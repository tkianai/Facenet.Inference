
import sys
sys.path.append('../')


from det_rec import build_det_model
import time
import torch


torch.set_grad_enabled(False)
model = build_det_model('../checkpoint/det_model.pth')
model.cuda()
model.eval()

batch_size = 32
image_size = 800

for _ in range(10):

    print("Start single image forward...")
    data = torch.Tensor(1, 3, image_size, image_size).cuda()
    torch.cuda.synchronize()
    _start = time.time()
    for i in range(batch_size):
        model(data)
    torch.cuda.synchronize()
    _end = time.time()
    print("Used: {}".format(_end - _start))

    print("Start batch image forward...")
    data = torch.Tensor(batch_size, 3, image_size, image_size).cuda()
    torch.cuda.synchronize()
    _start = time.time()
    model(data)
    torch.cuda.synchronize()
    _end = time.time()
    print("Used: {}".format(_end - _start))



'''
batch_size   |   image size   |   single/batch
16           |   1600         |    1.3/1.2
8            |   1600         |    0.64/0.6
8            |   800          |    0.19/0.15
16           |   800          |    0.37/0.31
32           |   800          |    0.75/0.66
'''
