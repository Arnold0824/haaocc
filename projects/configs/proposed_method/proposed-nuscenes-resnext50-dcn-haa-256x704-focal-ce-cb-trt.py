_base_ = ['./proposed-nuscenes-resnext50-dcn-haa-256x704-focal-ce-cb.py']

model = dict(
    wocc=True,
    wdet3d=False,
)
