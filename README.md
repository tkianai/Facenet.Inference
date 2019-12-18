
# Facenet.Inference

This repo using retinaface as detector, and face recognition model trained with arcface. It fixes the whole pipeline including detection, alignment and recognition.


## Usage

1. download the pretrained model, [det_model](https://drive.google.com/open?id=1Q0GowY1m4wVh2S1c7dZWzTxqBvb1sDej) and [rec_model](https://drive.google.com/open?id=17QstmDGSblJHXn_leZXOz2VyeKDYVIRN). Then put them into `./checkpoint` directory.
2. prepare the necessary data

- provide the facebank directory(`--facebank`), whose structure as bellow

```sh
id1
    id1_1.jpg
    id1_2.jpg
id2
    id2_1.jpg
...
```

- provide video file using `--vid`
- provide img file or directory using `--img`
- more details when running `python det_rec.py --help`

3. running the codes

```sh
python det_rec.py --vid <video path> --img <image path>
```

## Example

![example](./demo/1576669951085.gif)