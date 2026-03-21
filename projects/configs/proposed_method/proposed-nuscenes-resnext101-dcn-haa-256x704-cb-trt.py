_base_ = ['./proposed-nuscenes-resnext101-dcn-haa-256x704-cb.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)
