import os
import sys
import pdb

# inputs:
# [1]: the manifest input to contain meta data of the images
# [2]: path to folder that contains json files
print('python gen_manifest.py path_to_the_manifest_file path_to_folder_json_files')
print(sys.argv[1])

manifest_in = [f.rstrip().split(',') for f in open(sys.argv[1], 'r')]
#json_fol = '/data04/shared/hanle/prad_cancer_detection_SEER/data/heatmap_jsons_beatrice'
json_fol = sys.argv[2]

maps = {}
for path,studyid,subjectid,imageid,_ in manifest_in[1:]:
    slide_id = path.split('/')[-1].split('.')[0]
    json_path = 'heatmap_' + slide_id + '.json'
    if not os.path.exists(os.path.join(json_fol, json_path)):
        print(json_path + ' ...not exist')
        continue

    maps[slide_id] = [json_path, studyid, subjectid, imageid]

manifest = open(os.path.join(json_fol, 'manifest.csv'), 'w')
manifest.writelines('path,studyid,subjectid,imageid,"Image URL"\n')
for _, data in maps.items():
    json_path, studyid, subjectid, imageid = data
    manifest.writelines('./{},{},{},{}\n'.format(json_path, studyid, subjectid, imageid))

manifest.close()

