import copy
import csv
import functools
import glob
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

#delete later
#from model import SegmentationAugmentation
#from torch.utils.data import DataLoader
#from util.util import enumerateWithEstimate


raw_cache = getCache('part2ch13_raw')

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)#<3>
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('../LUNA_short/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    
    candidateInfo_list = []
    with open('../LUNA/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'0.0': False, '1.0': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True, #isNodule_bool
                    True, #hasAnnotation_bool
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )
    with open('../LUNA/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool: #only focused on non-nodules
                candidateInfo_list.append(CandidateInfoTuple(
                    False, #isNodule_bool
                    False, #hasAnnotation_bool
                    False, #isMal_bool
                    0.0, #candidateDiameter
                    series_uid,
                    candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    #print(candidateInfo_list)
    return candidateInfo_list


@functools.lru_cache(1) #<from7>
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)
    return candidateInfo_dict #<to7>
        


class Ct:
    def __init__(self, series_uid):
        #series_uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059'
        mhd_path = glob.glob(
                '../LUNA_short/subset*/{}.mhd'.format(series_uid)
            )[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        #ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        #<from6>
        candidateInfo_list = getCandidateInfoDict()[self.series_uid]
        
        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())
                            
        #<to6>
    

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool_)
        
        for candidateInfo_tup in positiveInfo_list:
            #<5from>
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
                )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)
            ### INDEX DIRECTION ###
            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                    self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1
            #<5to>
            ### ROW DIRECTION ###
            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                    self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1
            ### COLUMN DIRECTION ###
            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                    self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a 

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        #<from8>
        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]
        return ct_chunk, pos_chunk, center_irc
        #<to8>


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True) 
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                val_stride=0,
                isValSet_bool=None,
                series_uid=None,
                contextSlices_count=3,
                fullCt_bool = False,
            ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

 
        series_set_short = [x.split('/subset0/')[1][:-4] for x in glob.glob('../LUNA_short/subset*/*.mhd')]


        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        self.series_list = [x for x in self.series_list if x in series_set_short]
        test1 = self.series_list.copy()

        if isValSet_bool: #<1>
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = [] #<2>
        #print(self.series_list)
        for series_uid in self.series_list:
   
            index_count, positive_indexes = getCtSampleSize(series_uid)     
            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]
                
        self.candidateInfo_list = getCandidateInfoList() #<3>
        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]
        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))
       # print(self.series_list)


    def __len__(self):
        return len(self.sample_list)
        
    def __getitem__(self, ndx): #<4>
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)
        
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

            # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
            # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
            # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
            # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)
        return ct_t, pos_t, ct.series_uid, slice_ndx


class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #<1>

        self.candidateInfo_list = getCandidateInfoList() #<2>
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)
       

        ####CACHE LESS DATA######
        mhd_list = glob.glob('../LUNA_short/subset*/*.mhd')
        series_uid_short = [x.split('/')[-1][:-4] for x in mhd_list]
        self.candidateInfo_list = [candidate for candidate in self.candidateInfo_list if candidate[4] in series_uid_short]
        sample_filter = [candidate for candidate in self.candidateInfo_list if candidate[4] ==  '1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059']
        #print(self.candidateInfo_list)
        ###############

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96)) #<3>
        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)
        return 0, 1 #candidate_t, pos_t, series_uid, center_t




class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio_int = 2
            
    def __len__(self):
        return len(self.candidateInfo_list)
    
    
    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
       # print(candidateInfo_tup)
        return self.getitem_trainingCrop(candidateInfo_tup)
        
    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtRawCandidate( #<1>
        candidateInfo_tup.series_uid,
        candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3:4]

        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64,
                                       col_offset:col_offset+64]).to(torch.long)
        slice_ndx = center_irc.index
      #  print(slice_ndx)
        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx

    




#canInfo_list = copy.copy(getCandidateInfoList())
#pos_list = [ nt for nt in canInfo_list if not nt.isNodule_bool #<3>
        

#series_uid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059'
#ct_sample = getCt(series_uid)
#train_ds = TrainingLuna2dSegmentationDataset(
#            val_stride=10,
#            isValSet_bool=False,
#            contextSlices_count=3,
 #       )
#val_ds = Luna2dSegmentationDataset(
#            val_stride=10,
#            isValSet_bool=True,
#            contextSlices_count=3,
#        )


 

""""
train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

train_dl = DataLoader(
            train_ds,
            batch_size=16,
            num_workers=8,
            pin_memory= False,
        )
batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(0),
            start_ndx=train_dl.num_workers,
        )

train_ds_size = len(train_ds)
number_of_batches = len(train_dl)

augmentation_model = SegmentationAugmentation()
for batch_ndx, batch_tup in batch_iter:
    input_t, label_t, series_list, _slice_ndx_list = batch_tup
    input_g, label_g = augmentation_model(input_t, label_t)
    break







#test = ct_sample.buildAnnotationMask(pos_list)
#print(test)


"""