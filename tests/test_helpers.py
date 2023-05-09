from tidecv.helpers import json_to_Data, enlarge_dataset_to_respect_TIDE, filter_dataset_to_label
from tests.constants import TEST_ASSETS_DIR, mAP_threshold
from tidecv.quantify import TIDE


# Parse json to create the Data structures for GT and Pred
json_path = f"{TEST_ASSETS_DIR}/soda_df_box_20_images.json"
SODA_gts, SODA_preds = json_to_Data(json_path)

# Call TIDE on the entire Dataset
tide = TIDE(pos_threshold=mAP_threshold)
run = tide.evaluate(gt=SODA_gts, preds=SODA_preds, name="tide_run")


def test_enlarge_dataset_to_respect_TIDE():
    gts_keep = [0, 5, 10]
    preds_keep = [3, 11]

    gts_enlarged, preds_enlarged, gts_new_id_to_old_id, preds_new_id_to_old_id = enlarge_dataset_to_respect_TIDE(SODA_gts, SODA_preds, gts_keep, preds_keep)
    
    assert {gts_new_id_to_old_id[gt["_id"]] for gt in gts_enlarged.annotations} == {0, 5, 10, 17, 24}
    assert {preds_new_id_to_old_id[pred["_id"]] for pred in preds_enlarged.annotations} == {3, 11, 20, 24, 28}
    

def test_filter_dataset_to_label():
    cls_id = 1

    gts_filtered, preds_filtered = filter_dataset_to_label(SODA_gts, SODA_preds, cls_id)
    
    assert len(gts_filtered.annotations) == 134
    assert len(preds_filtered.annotations) == 133
