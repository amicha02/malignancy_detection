import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy as np 
import tqdm
import pylidc
import pandas as pd

annotations = pandas.read_csv('../LUNA/annotations.csv')
candidates = pd.read_csv('../LUNA/candidates.csv')
malignancy_data = []
missing = []
spacing_dict = {}
scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
suids = annotations.seriesuid.unique()

for suid in tqdm.tqdm(suids):
  #  print("First element of suids: {suid_zero}".format(suid_zero = suid))
    fn = glob.glob('../REPP/subset*/{}.mhd'.format(suid))
    if len(fn) == 0 or '*' in fn[0]:
        missing.append(suid)
        continue
    fn = fn[0]
   #  print("First element of fn: {fn_zero}".format(fn_zero = fn))
    x = sitk.ReadImage(fn)
    spacing_dict[suid] = x.GetSpacing()
    s = scans[suid]
    
    #There are 4 annotations per site. Here we iterate and average the annotations
    #for every nodule flagged as positive
    for ann_cluster in s.cluster_annotations():
        is_malignant = len([a.malignancy for a in ann_cluster if a.malignancy >= 4])>=2
        centroid = np.mean([a.centroid for a in ann_cluster], 0)
        bbox = np.mean([a.bbox_matrix() for a in ann_cluster], 0).T
        coord = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in centroid[[1, 0, 2]]])
        bbox_low = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[0, [1, 0, 2]]])
        bbox_high = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[1, [1, 0, 2]]])
        malignancy_data.append((suid, coord[0], coord[1], coord[2], bbox_low[0], bbox_low[1], bbox_low[2], bbox_high[0], bbox_high[1], bbox_high[2], is_malignant, [a.malignancy for a in ann_cluster]))





df_mal = pandas.DataFrame(malignancy_data, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ', 'mal_bool', 'mal_details'])

processed_annot = []
annotations['mal_bool'] = float('nan')
annotations['mal_details'] = [[] for _ in annotations.iterrows()]
bbox_keys = ['bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ']
for k in bbox_keys:
    annotations[k] = float('nan')
for series_id in tqdm.tqdm(annotations.seriesuid.unique()):
    #series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
    c = candidates[candidates.seriesuid == series_id]
    a = annotations[annotations.seriesuid == series_id]
    m = df_mal[df_mal.seriesuid == series_id]
    #print(m)
    if len(m) > 0:
        m_ctrs = m[['coordX', 'coordY', 'coordZ']].values
        a_ctrs = a[['coordX', 'coordY', 'coordZ']].values
        #print(m_ctrs.shape, a_ctrs.shape)
        matches = (np.linalg.norm(a_ctrs[:, None] - m_ctrs[None], ord=2, axis=-1) / a.diameter_mm.values[:, None] < 0.5)
        has_match = matches.max(-1)
        match_idx = matches.argmax(-1)[has_match]
        a_matched = a[has_match].copy()
        #c_matched['diameter_mm'] = a.diameter_mm.values[match_idx]
        a_matched['mal_bool'] = m.mal_bool.values[match_idx]
        a_matched['mal_details'] = m.mal_details.values[match_idx]
        for k in bbox_keys:
            a_matched[k] = m[k].values[match_idx]
        processed_annot.append(a_matched)
        processed_annot.append(a[~has_match])
    else:
        processed_annot.append(c)
processed_annot = pandas.concat(processed_annot)
processed_annot.sort_values('mal_bool', ascending=False, inplace=True)
processed_annot['len_mal_details'] = processed_annot.mal_details.apply(len)
df_nona = processed_annot.dropna()
df_nona.to_csv('annotations_with_malignancy.csv', index=False)




