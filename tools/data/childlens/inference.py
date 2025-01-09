from mmaction.apis.inferencers import MMAction2Inferencer


def main():
    # Define arguments as variables
    inputs = "/home/nele_pauline_suffo/ProcessedData/childlens_videos/106910.MP4"
    vid_out_dir = "/home/nele_pauline_suffo/outputs/bmn/inference"
    rec = "/home/nele_pauline_suffo/projects/mmaction2/configs/localization/bmn/bmn_2xb8-400x100-9e_childlens-feature.py"
    rec_weights = "/home/nele_pauline_suffo/outputs/bmn/best_auc_epoch_2.pth"
    label_file = "/home/nele_pauline_suffo/ProcessedData/childlens_annotations/action_name.csv"
    device = "cuda:0"  # Set to "cpu" or "cuda:0" depending on your hardware
    batch_size = 1
    show = False
    print_result = True
    pred_out_file = "/home/nele_pauline_suffo/outputs/bmn/inference/predictions"

    # Map variables to init_args and call_args
    init_args = {
        'rec': rec,
        'rec_weights': rec_weights,
        'device': device,
        'label_file': label_file
    }

    call_args = {
        'inputs': inputs,
        'vid_out_dir': vid_out_dir,
        'batch_size': batch_size,
        'show': show,
        'print_result': print_result,
        'pred_out_file': pred_out_file
    }

    # Run inference
    mmaction2 = MMAction2Inferencer(**init_args)
    mmaction2(**call_args)


if __name__ == '__main__':
    main()