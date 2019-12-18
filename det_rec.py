
import os
import os.path as osp
import argparse
import cv2
from PIL import Image
import torch
from detector import RetinaFace, PriorBox
from detector import utils as det_utils
from aligner import align_face
from recognizer import Arcface
from recognizer import utils as rec_utils
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Face recognition pipeline')
    parser.add_argument('--vid', type=str, default=None, help='input video')
    parser.add_argument('--img', type=str, default=None, help='input image or directory')
    parser.add_argument('--facebank', type=str, default='./data/facebank', help='facebank directory')
    parser.add_argument('--update_facebank', action="store_true", default=False, help='update facebank')
    parser.add_argument('--down_sample', type=float, default=1, help='down sample the origin image')
    parser.add_argument('--cpu', action="store_true", default=False, help='use cpu inference')
    parser.add_argument('--save', type=str, default='./results', help='output directory')

    args = parser.parse_args()
    assert (args.vid is not None or args.img is not None), 'Nothing input!'

    if not osp.exists(args.save):
        os.makedirs(args.save)

    return args


def build_det_model(ckpt_path='./checkpoint/det_model.pth'):
    det_model = RetinaFace()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    det_model.load_state_dict(checkpoint)
    return det_model


def build_rec_model(ckpt_path='./checkpoint/rec_model.pth'):
    rec_model = Arcface()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    rec_model.load_state_dict(checkpoint)
    return rec_model


def is_valid_file(x, extens=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
    return x.lower().endswith(extens)


def prepare_facebank(det_model, rec_model, dir_path, device=torch.device('cpu'), update=False):
    if not update and osp.exists(osp.join(dir_path, 'facebank.pth')):
        facebank = torch.load(osp.join(dir_path, 'facebank.pth'))
    else:
        embeddings = []
        classes = []
        names = [d.name for d in os.scandir(dir_path) if d.is_dir()]
        for name in names:
            embs = []
            for imgname in os.listdir(osp.join(dir_path, name)):
                if not is_valid_file(imgname):
                    continue
                img_raw = cv2.imread(osp.join(dir_path, name, imgname))
                output = inference(det_model, img_raw, rec_model=rec_model, return_embedding=True, device=device)
                assert len(output) == 1, "facebank is not clean, including many person in one picture!"
                embs.append(output[0]['feature'])
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            classes.append(name)
        classes.append('Unknown')
        classes = np.array(classes)
        facebank = (classes, torch.cat(embeddings))
        torch.save(facebank, osp.join(dir_path, 'facebank.pth'))

    return facebank


def inference(det_model, img_raw, rec_model=None, facebank=None, device=torch.device('cpu'), det_thresold=0.9, top_k=100, nms_threshold=0.4, rec_threshold=1.6, rec_tta=True, return_embedding=False):
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    loc, conf, landms = det_model(img)  # forward pass

    priorbox = PriorBox(image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = det_utils.decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = det_utils.decode_landm(
        landms.data.squeeze(0), prior_data, [0.1, 0.2])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > det_thresold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = det_utils.py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # align face
    output = []
    for i, det in enumerate(dets):
        output_item = {
            'score': det[4].item(),
            'bbox': list(map(int, det[:4].tolist())),
        }

        if rec_model is not None:
            face_pts = landms[i].reshape(5, 2)
            face = align_face(img_raw, face_pts)

            # recognize
            # resize image to [128, 128]
            resized = cv2.resize(face, (128, 128))

            # center crop image
            a = int((128-112)/2)  # x start
            b = int((128-112)/2+112)  # x end
            c = int((128-112)/2)  # y start
            d = int((128-112)/2+112)  # y end
            ccropped = resized[a:b, c:d]  # center crop the image
            ccropped = ccropped[..., ::-1]  # BGR to RGB

            # load numpy to tensor
            ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
            ccropped = np.reshape(ccropped, [1, 3, 112, 112])
            ccropped = np.array(ccropped, dtype=np.float32)
            ccropped = (ccropped - 127.5) / 128.0
            ccropped = torch.from_numpy(ccropped)

            feature = rec_model(ccropped.to(device))
            if rec_tta:
                flipped = torch.flip(ccropped, [3])
                feature += rec_model(flipped.to(device))
            feature = rec_utils.l2_norm(feature)

            if return_embedding:
                output_item.update({'feature': feature})
        
        if facebank is not None:

            # classify
            source_embeddings = feature
            classes, target_embeddings = facebank
            target_embeddings = target_embeddings.to(device)
            diff = source_embeddings.unsqueeze(-1) - target_embeddings.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)
            min_idx[minimum > rec_threshold] = -1  # if no match, set idx to -1
            name = classes[min_idx.cpu().numpy()]

            output_item.update({'name': name[0]})

        output.append(output_item)

    return output


def main(args):
    
    torch.set_grad_enabled(False)
    # define detector model
    det_model = build_det_model()
    # define recognize model
    rec_model = build_rec_model()
    # eval model
    det_model.eval()
    rec_model.eval()

    # running device
    device = torch.device('cpu' if args.cpu else 'cuda')
    det_model.to(device)
    rec_model.to(device)

    # facebank
    facebank = prepare_facebank(det_model, rec_model, args.facebank, device=device, update=args.update_facebank)

    if args.vid is not None:
        vc = cv2.VideoCapture(args.vid)
        vc.isOpened()

        # write to video
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vc.get(cv2.CAP_PROP_FPS))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) / args.down_sample)
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) / args.down_sample)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = osp.join(args.save, osp.basename(args.vid).split('.')[0] + '.mp4')
        print(save_path)
        vw = cv2.VideoWriter(save_path, fourcc, fps, size)

        pbar = tqdm(total=total_frames)
        while True:
            flag, frame = vc.read()
            if not flag:
                break
            frame = cv2.resize(frame, (width, height))
            output = inference(det_model, frame, rec_model=rec_model, facebank=facebank, device=device)

            # visualize
            for _item in output:
                bbox = _item['bbox']
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                descript = "{}: {:.3f}".format(_item['name'], _item['score'])
                cx = bbox[0]
                cy = bbox[1] + 12
                cv2.putText(frame, descript, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # save result
            vw.write(frame)

            pbar.update(1)
        vc.release()
        vw.release()

    if args.img is not None:
        images = []
        if osp.isdir(args.img):
            for imagename in os.listdir(args.img):
                images.append(osp.join(args.img, imagename))
        else:
            images.append(args.img)

        for img_path in tqdm(images):
            img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_raw = cv2.resize(img_raw, fx=1.0/args.down_sample, fy=1.0/args.down_sample)
            output = inference(det_model, img_raw, rec_model=rec_model, facebank=facebank, device=device)

            # visualize
            for _item in output:
                bbox = _item['bbox']
                cv2.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                descript = "{}: {:.3f}".format(_item['name'], _item['score'])
                cx = bbox[0]
                cy = bbox[1] + 12
                cv2.putText(img_raw, descript, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # save result
            basename = osp.basename(img_path)
            cv2.imwrite(osp.join(args.save, basename), img_raw)


if __name__ == "__main__":
    args = parse_args()
    main(args)
