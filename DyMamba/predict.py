import os
join = os.path.join
import argparse
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
from baseline.models.unetr2d import UNETR2D
import time
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)



def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='/data/cellseg/images', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/data/outpus/dymamba/cellseg', type=str, help='output path')
    parser.add_argument('--model_path',default="", type=str, help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='dymamba')
    parser.add_argument('--num_class', default=2, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')

    # wojiade
    parser.add_argument("--device", default="cuda:1", type=str, choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], help="select device to run the model")
    parser.add_argument("--dataset", default="kavsir", type=str, choices=["lucchi", "kavsir", "cellpose", "mitoem", "nips", "livecell"])

    # 解析
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dataset.lower() == "lucchi":
        args.data_path = "/data/Lucchi"
    if args.dataset.lower() == "mitoem":
        args.data_path = "/data/MitoEMR"
    if args.dataset.lower() == "kavsir":
        args.data_path = "/data/Kvasir-SEG"
    if args.dataset.lower() == "nips":
        args.data_path = "/data/nips"
    if args.dataset.lower() == "cellpose":
        args.data_path = "/data/cellpose"
    if args.dataset.lower() == "livecell":
        args.data_path = "/data/livecell"

    input_path = os.path.join(args.data_path, 'images')
    output_path = os.path.join(args.output_path, args.model_name, args.dataset)

    print('input_path: ', input_path)
    print('output_path: ', output_path)

    print("model_path", args.model_path)



    os.makedirs(output_path, exist_ok=True)

    img_names = sorted(os.listdir(join(input_path)))

    model= None

    if args.model_name.lower() == "dymamba":
        print('prepare to load dymamba success')
        from vmamba.DynamicM import DyMamba
        vss_args = dict(
            in_chans=3, 
            patch_size=4, 
            depths=[2,2,4,2], 
            dims=96, 
            drop_path_rate=0.2)
        decoder_args = dict(
            num_classes=args.num_class,
            deep_supervision=False, 
            features_per_stage=[96, 192, 384, 768],      
            drop_path_rate=0.2,
            d_state=16)
        model = DyMamba(vss_args, decoder_args).to(device)
        print('loading dymamba success')


    checkpoint = torch.load(join(args.model_path, 'best_Dice_model.pth'), map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                print('input_path, img_name: ',(input_path, img_name))
                img_data = io.imread(join(input_path, img_name))
            
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            test_pred_npy = test_pred_out[0,1].cpu().numpy()
            # convert probability map to binary mask and apply morphological postprocessing
            test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>0.5),16))
            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            
        
if __name__ == "__main__":
    main()









