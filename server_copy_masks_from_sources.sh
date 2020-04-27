TARGET_DP="/media/data10T_1/datasets/CRDF_5_tmp/model_dataset/original/masks"
MASKS_ORIENTATION_FIXED_BIN_DP="/media/data10T_1/datasets/CRDF_5_tmp/masks_resegm2_old/masks_orientation_fixed_binary"
MASKS_AUTOLUNGS_FIXED_DP="/media/data10T_1/datasets/CRDF_5_tmp/masks_autolungs_to_fix/masks_autolungs_fixed"

# create dir if needed
if [ ! -d ${TARGET_DP} ]; then
  mkdir ${TARGET_DP}
fi
cp ${MASKS_ORIENTATION_FIXED_BIN_DP}/*.nii.gz ${TARGET_DP}
cp ${MASKS_AUTOLUNGS_FIXED_DP}/*.nii.gz ${TARGET_DP}
