_base_ = ['./ablation-nuscenes-resnext101-dcn-256x704-focal-ce-cb.py',
          ]

model = dict(
    wocc=True,
    wdet3d=False,
)
