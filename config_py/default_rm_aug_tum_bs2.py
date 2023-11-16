seed = 12
deterministic = False
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

log_config = dict(
    interval=2,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", init_kwargs={"project": "OBJMAP_bs1_gpu2"}),
        dict(type="CheckpointHook", interval=200, by_epoch=False, max_keep_ckpts=6),
    ],
)

load_from = None
resume_from = None

cudnn_benchmark = False
fp16 = dict(loss_scale=dict(growth_interval=2000))
max_epochs = 20
runner = dict(type="CustomEpochBasedRunner", max_epochs=max_epochs)


dataset_type = "NuScenesDataset"
dataset_root = "data/nuscenes/"
gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
canvas_size = (100, 50)
image_size = [256, 704]
map_head_dim = 256

augment2d = dict(
    resize=[[1, 1], [0.48, 0.48]],
    rotate=[-0.0, 0.0],
    gridmask=dict(prob=0.0, fixed_prob=True),
)
map_class2label = {
    "ped_crossing": 0,
    "divider": 1,
    "contours": 2,
    "others": -1,
}
map_num_class = max(list(map_class2label.values())) + 1

augment3d = dict(scale=[0.9, 1.1], rotate=[-0.78539816, 0.78539816], translate=0.5)


object_classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


map_classes = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
]


input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)

#! train_pipeline
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    # dict(
    #     type="ObjectPaste",
    #     stop_epoch=gt_paste_stop_epoch,
    #     db_sampler=dict(
    #         dataset_root=dataset_root,
    #         info_path=dataset_root + "nuscenes_dbinfos_train.pkl",
    #         rate=1.0,
    #         prepare=dict(
    #             filter_by_difficulty=[-1],
    #             filter_by_min_points=dict(
    #                 car=5,
    #                 truck=5,
    #                 bus=5,
    #                 trailer=5,
    #                 construction_vehicle=5,
    #                 traffic_cone=5,
    #                 barrier=5,
    #                 motorcycle=5,
    #                 bicycle=5,
    #                 pedestrian=5,
    #             ),
    #         ),
    #         classes=object_classes,
    #         sample_groups=dict(
    #             car=2,
    #             truck=3,
    #             construction_vehicle=7,
    #             bus=4,
    #             trailer=6,
    #             barrier=2,
    #             motorcycle=6,
    #             bicycle=6,
    #             pedestrian=2,
    #             traffic_cone=2,
    #         ),
    #         points_loader=dict(
    #             type="LoadPointsFromFile",
    #             coord_type="LIDAR",
    #             load_dim=load_dim,
    #             use_dim=use_dim,
    #             reduce_beams=reduce_beams,
    #         ),
    #     ),
    # ),
    dict(
        type="ImageAug3D",  # 会添加 img_aug_matrix 参数
        final_dim=image_size,
        resize_lim=augment2d["resize"][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d["rotate"],
        rand_flip=False,
        is_train=False,
    ),
    dict(
        type="GlobalRotScaleTrans",  # 会添加 lidar_aug_matrix 参数
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False,
    ),
    dict(
        type="LoadBEVSegmentation",
        dataset_root=dataset_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes,
    ),
    # dict(type="RandomFlip3D"), # 会修改 lidar_aug_matrix 参数
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=object_classes),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="GridMask",
        use_h=True,
        use_w=True,
        max_epoch=max_epochs,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=augment2d["gridmask"]["prob"],
        fixed_prob=augment2d["gridmask"]["fixed_prob"],
    ),
    dict(type="PointShuffle"),
    dict(
        type="VectorizeLocalMap",
        data_root="./data/nuscenes",
        patch_size=[60, 30],
        sample_dist=0.7,
        num_samples=150,
        sample_pts=False,
        max_len=30,
        padding=False,
        normalize=True,
        fixed_num=dict(ped_crossing=-1, divider=-1, contours=-1, others=-1),
        class2label=map_class2label,
        centerline=False,
    ),
    dict(
        type="PolygonizeLocalMapBbox",
        canvas_size=canvas_size,  # xy
        coord_dim=2,
        num_class=3,
        mode="xyxy",
        test_mode=True,
        threshold=0.02,
        flatten=False,
    ),
    dict(type="FormatBundleMap"),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(type="SkipSample"),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d", "gt_masks_bev", "polys"],
        meta_keys=[
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "camera2lidar",
            "lidar2image",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d["resize"][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d["rotate"],
        rand_flip=False,
        is_train=False,
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False,
    ),
    dict(
        type="LoadBEVSegmentation",
        dataset_root=dataset_root,
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        classes=map_classes,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(type="SkipSample"),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d", "gt_masks_bev"],
        meta_keys=[
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "camera2lidar",
            "lidar2image",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            object_classes=object_classes,
            map_classes=map_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

evaluation = dict(interval=1000, pipeline=test_pipeline)

#! MODEL
encoder_camera_backbone = dict(
    type="SwinTransformer",
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    patch_norm=True,
    out_indices=[1, 2, 3],
    with_cp=False,
    convert_weights=True,
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    ),
)
encoder_camera_neck = dict(
    type="GeneralizedLSSFPN",
    in_channels=[192, 384, 768],
    out_channels=256,
    start_level=0,
    num_outs=3,
    norm_cfg=dict(type="BN2d", requires_grad=True),
    act_cfg=dict(type="ReLU", inplace=True),
    upsample_cfg=dict(mode="bilinear", align_corners=False),
)
encoder_camera_vtransform = dict(
    type="DepthLSSTransform",
    in_channels=256,
    out_channels=80,
    image_size=image_size,
    feature_size=[image_size[0] // 8, image_size[1] // 8],
    xbound=[-54.0, 54.0, 0.3],
    ybound=[-54.0, 54.0, 0.3],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 0.5],
    downsample=2,
)
encoder_lidar_voxelize = dict(
    max_num_points=10,
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    max_voxels=[120000, 160000],
)
encoder_lidar_backbone = dict(
    type="SparseEncoder",
    in_channels=5,
    sparse_shape=[1440, 1440, 41],
    output_channels=128,
    order=["conv", "norm", "act"],
    encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
    encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
    block_type="basicblock",
)

decoder_backbone = dict(
    type="SECOND",
    in_channels=256,
    out_channels=[128, 256],
    layer_nums=[5, 5],
    layer_strides=[1, 2],
    norm_cfg=dict(type="BN", eps=1.0e-3, momentum=0.01),
    conv_cfg=dict(type="Conv2d", bias=False),
)
decoder_neck = dict(
    type="SECONDFPN",
    in_channels=[128, 256],
    out_channels=[256, 256],
    upsample_strides=[1, 2],
    norm_cfg=dict(type="BN", eps=1.0e-3, momentum=0.01),
    upsample_cfg=dict(type="deconv", bias=False),
    use_conv_for_no_stride=True,
)

object_heads = dict(
    type="TransFusionHead",
    num_proposals=200,
    auxiliary=True,
    in_channels=512,
    hidden_channel=128,
    num_classes=10,
    num_decoder_layers=1,
    num_heads=8,
    nms_kernel_size=3,
    ffn_channel=256,
    dropout=0.1,
    bn_momentum=0.1,
    activation="relu",
    train_cfg=dict(
        dataset="nuScenes",
        point_cloud_range=point_cloud_range,
        grid_size=[1440, 1440, 41],
        voxel_size=voxel_size,
        out_size_factor=8,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3D",
            iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
            cls_cost=dict(type="FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
            reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
            iou_cost=dict(type="IoU3DCost", weight=0.25),
        ),
    ),
    test_cfg=dict(
        dataset="nuScenes",
        grid_size=[1440, 1440, 41],
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        pc_range=point_cloud_range[:2],
        nms_type=None,
    ),
    common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
    bbox_coder=dict(
        type="TransFusionBBoxCoder",
        pc_range=point_cloud_range[:2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        score_threshold=0.0,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=10,
    ),
    loss_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        loss_weight=1.0,
    ),
    loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean", loss_weight=1.0),
    loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
)

vectormap_heads = dict(
    type="DGHead",
    augmentation=True,
    augmentation_kwargs=dict(
        p=0.3,
        scale=0.01,
        bbox_type="xyxy",
    ),
    det_net_cfg=dict(
        type="MapElementDetector",
        num_query=120,
        max_lines=35,
        bbox_size=2,
        canvas_size=canvas_size,
        separate_detect=False,
        discrete_output=False,
        num_classes=map_num_class,
        in_channels=512,
        score_thre=0.1,
        num_reg_fcs=2,
        num_points=4,
        iterative=False,
        pc_range=[-15, -30, -5.0, 15, 30, 3.0],
        sync_cls_avg_factor=True,
        transformer=dict(
            type="DeformableDetrTransformer_",
            encoder=dict(
                type="PlaceHolderEncoder",
                embed_dims=map_head_dim,
            ),
            decoder=dict(
                type="DeformableDetrTransformerDecoder_",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=map_head_dim,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                            dropout_layer=dict(type="Dropout", drop_prob=0.1),
                        ),
                        # dict(
                        #     type='MultiScaleDeformableAttentionFp16',
                        #     init_cfg = None,
                        #     attn_cfg = dict(
                        #             type='MultiScaleDeformableAttention',
                        #             embed_dims=map_head_dim,
                        #             num_heads=8,
                        #             num_levels=1,
                        #         )
                        #     ), # MultiScaleDeformableAttentionFP32
                        dict(
                            type="MultiScaleDeformableAttentionFP32",
                            embed_dims=map_head_dim,
                            num_heads=8,
                            num_levels=1,
                        ),
                        # dict(
                        #         type='MultiScaleDeformableAttention',
                        #         embed_dims=map_head_dim,
                        #         num_heads=8,
                        #         num_levels=1,
                        #     ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=map_head_dim,
                        feedforward_channels=map_head_dim * 2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    feedforward_channels=map_head_dim * 2,
                    ffn_dropout=0.1,
                    operation_order=(
                        "norm",
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding",
            num_feats=map_head_dim // 2,
            normalize=True,
            offset=-0.5,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_reg=dict(type="LinesLoss", loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(
                type="HungarianLinesAssigner",
                cost=dict(
                    type="MapQueriesCost",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBoxCostC", weight=0.1),  # continues
                    iou_cost=dict(
                        type="IoUCostC", weight=1, box_format="xyxy"
                    ),  # continues
                ),
            ),
        ),
    ),
    gen_net_cfg=dict(
        type="PolylineGenerator",
        in_channels=512,
        encoder_config=None,
        decoder_config={
            "layer_config": {
                "d_model": 256,
                "nhead": 8,
                "dim_feedforward": 512,
                "dropout": 0.2,
                "norm_first": True,
                "re_zero": True,
            },
            "num_layers": 6,
        },
        class_conditional=True,
        num_classes=map_num_class,
        canvas_size=canvas_size,  # xy
        max_seq_length=500,
        decoder_cross_attention=False,
        use_discrete_vertex_embeddings=True,
    ),
    max_num_vertices=80,
    top_p_gen_model=0.9,
    sync_cls_avg_factor=True,
)

fuser = dict(type="ConvFuser", in_channels=[80, 256], out_channels=256)
model = dict(
    type="BEVFusionMap",
    encoders=dict(
        camera=dict(
            backbone=encoder_camera_backbone,
            neck=encoder_camera_neck,
            vtransform=encoder_camera_vtransform,
        ),
        lidar=dict(voxelize=encoder_lidar_voxelize, backbone=encoder_lidar_backbone),
    ),
    fuser=fuser,
    decoder=dict(backbone=decoder_backbone, neck=decoder_neck),
    heads=dict(object=object_heads, vectormap=vectormap_heads),
)

optimizer = dict(type="AdamW", lr=2.0e-4, weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1.0e-3,
)
momentum_config = dict(policy="cyclic")
