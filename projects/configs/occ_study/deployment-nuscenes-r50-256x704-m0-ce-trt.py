_base_ = ['./main-nuscenes-r50-256x704-m0-ce.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)
