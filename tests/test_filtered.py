from collections import defaultdict
import random

from tidecv.quantify import TIDE
from tests.constants import TEST_ASSETS_DIR, mAP_threshold, RANDOM_SEED
from tidecv.helpers import json_to_Data, enlarge_dataset_to_respect_TIDE, create_filtered_Data


# Parse json to create the Data structures for GT and Pred
json_path = f"{TEST_ASSETS_DIR}/soda_df_box_20_images.json"
SODA_gts, SODA_preds = json_to_Data(json_path)

# Call TIDE on the entire Dataset
tide = TIDE(pos_threshold=mAP_threshold)
run = tide.evaluate(gt=SODA_gts, preds=SODA_preds, name="tide_run")


def test_recalc_on_filtered():
    """
    Test that recalculating mAP/impact on mAP/errors on any filtered dataset can be done
    from the original run tide object (without recalculating TIDE from scratch) 
    """
    # Select 50 GTs and 50 Preds, and create the filtered dataset with all added links
    random.seed(RANDOM_SEED)
    gts_keep = set(random.sample([ann["_id"] for ann in SODA_gts.annotations], 5))
    preds_keep = set(random.sample([ann["_id"] for ann in SODA_preds.annotations], 5))

    SODA_gts_filtered, SODA_preds_filtered, gts_enlarged_new_id_to_old_id, preds_enlarged_new_id_to_old_id = enlarge_dataset_to_respect_TIDE(SODA_gts, SODA_preds, gts_keep, preds_keep)

    # Calculate TIDE on the filtered (+ enlarged) data
    tide_filtered = TIDE(pos_threshold=mAP_threshold)
    run_filtered = tide_filtered.evaluate(gt=SODA_gts_filtered, preds=SODA_preds_filtered, name="tide_run_filtered")
    
    APs_filtered = run_filtered.ap_data.get_APs()
    mAP_filtered = run_filtered.ap_data.get_mAP()
    # errors_filtered = tide_filtered.get_main_errors()["tide_run_filtered"]


    # Calculate TIDE by restricting to ids
    # First need to get a dict cls -> ids for both GT and Pred
    SODA_gts_filtered_cls_to_ids = defaultdict(set)
    for ann in SODA_gts_filtered.annotations:
        SODA_gts_filtered_cls_to_ids[ann["class"]].add(gts_enlarged_new_id_to_old_id[ann["_id"]])
    SODA_preds_filtered_cls_to_ids = defaultdict(set)
    for ann in SODA_preds_filtered.annotations:
        SODA_preds_filtered_cls_to_ids[ann["class"]].add(preds_enlarged_new_id_to_old_id[ann["_id"]])

    errors_restricted = tide.get_main_errors(pred_dict=SODA_preds_filtered_cls_to_ids, gt_dict=SODA_gts_filtered_cls_to_ids)["filtered_errors_in_run_tide_run"]
    APs_restricted = tide.runs['tide_run'].qualifiers['restricted_APs']
    mAP_restricted = tide.runs['tide_run'].qualifiers['restricted_mAP']


    # Assert that mAP on filtered/enlarged = restricted TIDE on original
    assert mAP_filtered == mAP_restricted

    # Assert that APs on filtered/enlarged = restricted TIDE on original
    assert APs_filtered == APs_restricted

    # Assert that impact on mAP on filtered/enlarged = restricted TIDE on original
    # TODO
