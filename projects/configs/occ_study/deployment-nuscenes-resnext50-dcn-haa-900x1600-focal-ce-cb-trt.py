_base_ = ['./main-nuscenes-resnext50-dcn-haa-900x1600-focal-ce-cb.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)
