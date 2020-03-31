from utils import *
from FourPanelGleasonImage import FourPanelGleasonImage
from HeatMap import HeatMap
from MergedHeatMap import MergedHeatMap
from PredictionFile import  PredictionFile
from StagedTumorHeatMap import StagedTumorHeatMap



# these folders will be replaced by paramaters
svs_fol_staged = '/data10/shared/hanle/svs_SEER_PRAD'
staged_pred = '/data04/shared/hanle/prad_cancer_detection_SEER/data/heatmap_txt_3classes_header_seer1'
output_pred = '/data02/shared/hanle/test_4panedl_seer_prad'

# test_svs_id = 'TCGA-2A-A8VO-01Z-00-DX1'

prefix = "prediction-"
wsi_extension = ".tif"
skip_first_line_pred = True
args = None


def checkFileExisting(wsiId):
    allPath = [
        os.path.join(staged_pred, 'color-'+wsiId), # colorPath
        os.path.join(svs_fol_staged, wsiId+wsi_extension), # svsPath
        os.path.join(staged_pred, prefix+wsiId), # predPath
        #os.path.join(til_fol, 'prediction-'+wsiId), # tilPath_pred
        #os.path.join(til_fol, 'color-'+wsiId), # tilPath_color
    ]
    ans = True
    for path in allPath:
        ans = os.path.exists(path)
        if not ans:
            print(path, "does not exit!")
            break
    return ans


def gen1Image(f):
    wsiId = f[len(prefix):]
    if not checkFileExisting(wsiId):
        return
    oslide = openslide.OpenSlide(os.path.join(svs_fol_staged, wsiId+wsi_extension))
    stagedCancerFile = StagedTumorHeatMap(staged_pred, skip_first_line_pred)
    stagedCancerFile.setWidthHeightByOSlide(oslide)
    stagedCancerMap = stagedCancerFile.getHeatMapByID(wsiId)
    classificationMap = stagedCancerFile.getTumorClassificationMap()
    stageClassificationMap = stagedCancerFile.getStageClassificationMap()

    img = FourPanelGleasonImage(oslide, stagedCancerMap, classificationMap, stageClassificationMap,
                       os.path.join(output_pred, wsiId+".png"))
    img.saveImg()
    print(wsiId)


def main(parallel_processing = False):
    if not os.path.isdir(output_pred):
        os.makedirs(output_pred)
    print(prefix)

    grades_prediction_fns = [f for f in os.listdir(staged_pred) if f.startswith(prefix)]

    if len(grades_prediction_fns) == 0:
        print("No valid file!")
        return

    if not parallel_processing:
        for i, f in enumerate(grades_prediction_fns):
            gen1Image(f)
            print(i, f)
    else:
        num_of_cores = multiprocessing.cpu_count() - 2
        p = multiprocessing.Pool(num_of_cores)
        p.map(gen1Image, grades_prediction_fns)

if __name__ == "__main__":
    is_parallel = True if str2bool(sys.argv[1]) else False
    main(parallel_processing = is_parallel)




