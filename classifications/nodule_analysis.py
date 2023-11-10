import argparse
import glob
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate

from segmentation.dsets import Luna2dSegmentationDataset
from dsets import LunaDataset,  getCt, getCandidateInfoDict, getCandidateInfoList, CandidateInfoTuple
from segmentation.model import UNetWrapper
from model import LunaModel

from util.logconf import logging
from util.util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
logging.getLogger("segmentation.dsets").setLevel(logging.WARNING)
logging.getLogger("classifications.dsets").setLevel(logging.WARNING)


def match_and_score(detections, truth, threshold=0.5, threshold_mal=0.5):
    # Returns 3x4 confusion matrix for:
    # Rows: Truth: Non-Nodules, Benign, Malignant
    # Cols: Not Detected, Detected by Seg, Detected as Benign, Detected as Malignant
    # If one true nodule matches multiple detections, the "highest" detection is considered
    # If one detection matches several true nodule annotations, it counts for all of them
    true_nodules = [c for c in truth if c.isNodule_bool]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    detected_xyz = np.array([n[2] for n in detections])
    # detection classes will contain
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
    # 3 -> detected as malignant nodule (if applicable)
    detected_classes = np.array([1 if d[0] < threshold
                                 else (2 if d[1] < threshold
                                       else 3) for d in detections])

    confusion = np.zeros((3, 4), dtype=np.int)
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion[2 if tn.isMal_bool else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    else:
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:, None]
        matches = (normalized_dists < 0.7)
        unmatched_detections = np.ones(len(detections), dtype=np.bool)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int)
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1
    return confusion
def print_confusion(label, confusions, do_mal):
    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))


class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            #log.debug(sys.argv)
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='models/segmentation/seg_2023-11-04_16.01.59_none.best.state',
        )

        parser.add_argument('--cls-model',
            help="What to model class name to use for the classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
          #  default = 'models/classification/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state'
            default = 'models/classification/cls_2023-11-08_17.43.18_AM.500.state'
        )

        parser.add_argument('--malignancy-model',
            help="What to model class name to use for the malignancy classifier.",
            action='store',
            default='LunaModel',
            # default='ModifiedLunaModel',
        )
        parser.add_argument('--malignancy-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--tb-prefix',
            default='ts_classification',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('series_uid',
            nargs='?',
            default="1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260",
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("One and only one of series_uid and --run-validation should be given")
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is MPS available? {torch.backends.mps.is_available()}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == 'mps':
            self.device = torch.device(self.device)
            self.use_mps = True
            self.use_cuda = False


        print(f"Using device: {self.device}")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.initModelPath('seg')
        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initModelPath('cls')

        self.seg_model, self.cls_model, self.malignancy_model = self.initModels()
    
    def initModels(self):
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()
        seg_model.to(self.device)

       # if False:
        if self.cli_args.classification_path:
            log.debug(self.cli_args.classification_path)
            cls_model = LunaModel()
            cls_dict = torch.load(self.cli_args.classification_path, map_location=torch.device('mps'))     
  
            cls_model.load_state_dict(cls_dict['model_state'])
            cls_model.eval()
           # cls_model.to(self.device)
        else:
            cls_model = None

        if self.cli_args.malignancy_path:
            malignancy_model = LunaModel()
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None


        return seg_model, cls_model, malignancy_model

    def initModelPath(self, type_str):
        local_path = os.path.join(
            'models',
            type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(

                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise
    

    
    def initSegmentationDl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
                contextSlices_count=3,
                series_uid=series_uid,
                fullCt_bool=True,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size ,
            num_workers=self.cli_args.num_workers,
            pin_memory=0,
        )

        return seg_dl
    

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )  #<1>
        val_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in val_ds.candidateInfo_list
        )
        positive_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in getCandidateInfoList()
            if candidateInfo_tup.isNodule_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in getCandidateInfoList()
            )

        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []
        val_list = sorted(series_set & val_set)
      

        candidateInfo_dict = getCandidateInfoDict()
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )

        all_confusion = np.zeros((3, 4), dtype=np.int)

        for _, series_uid in series_iter:
            ct = getCt(series_uid) #<2>
            mask_a = self.segmentCt(ct, series_uid) #<3>
            candidateInfo_list = self.groupSegmentationOutput(
                series_uid, ct, mask_a) #<4>
           
            classifications_list = self.classifyCandidates(
                ct, candidateInfo_list) #<5>
            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5: 
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)
            if series_uid in candidateInfo_dict: #<6>
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                )
                print(one_confusion)
                all_confusion += one_confusion
                print_confusion(series_uid, one_confusion, self.malignancy_model is not None) #<a>
                print_confusion("Total", all_confusion, self.malignancy_model is not None)
      





    def initClassificationDl(self, candidateInfo_list):
        cls_ds = LunaDataset(
                sortby_str='series_uid',
                candidateInfo_list=candidateInfo_list,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return cls_dl
    





    def classifyCandidates(self, ct, candidateInfo_list):
        cls_dl = self.initClassificationDl(candidateInfo_list)
        classifications_list = []
        for batch_ndx, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            #input_g = input_t.to(self.device)
            with torch.no_grad():
                _, probability_nodule_g = self.cls_model(input_t)
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_t)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            zip_iter = zip(center_list,
                probability_nodule_g[:,1].tolist(),
                probability_mal_g[:,1].tolist())
            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = irc2xyz(center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxSize_xyz=ct.vxSize_xyz,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup) #<b>
        return classifications_list








            
    def groupSegmentationOutput(self, series_uid,  ct, clean_a):
        candidateLabel_a, candidate_count = measurements.label(clean_a) #<a>
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count+1),
        )  #<b>

        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            candidateInfo_tup = \
                CandidateInfoTuple(False, False, False, 0.0, series_uid, center_xyz) #<c> 
            candidateInfo_list.append(candidateInfo_tup)
        
        return candidateInfo_list
    


    def segmentCt(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.initSegmentationDl(series_uid)  #  <a>
            for input_t, _, _, slice_ndx_list in seg_dl:
                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g) #<b>
                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1) #<c>
        return mask_a


if __name__ == '__main__':
    NoduleAnalysisApp().main()
