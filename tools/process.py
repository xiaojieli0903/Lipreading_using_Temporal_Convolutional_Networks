import os
import shutil

items = os.listdir('./')
target = 'frontend3D_nomixup_20epoch_predict-type1-'
target = 'usememory_weight1_avg0_detach_'
target = 'pf1_recon1_constrastive1_fixmemory_'
#target = 'mem'
target = 'cosine_mem_addloss_constrastive_fixmemory_'
#replace = 'mvm'
replace = ''
for item in items:
    if item.find(target) >= 0:
        shutil.move(item, item.replace(target, replace))
