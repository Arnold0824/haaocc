_base_ = ['./main-nuscenes-r50-900x1600-focal-ce.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)
