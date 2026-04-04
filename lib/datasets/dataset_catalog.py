from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'SbdTrain': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'SbdVal': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'SbdMini': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'mini'
        },
        'SbdMedicalTrain': {
            'id': 'sbd',
            'data_root': '/mnt/sdb1/leijh/EnergySnake1/Data_processed/1232processed',
            'ann_file': 'dummy_annotations.json',
            'split': 'mini'
        },
        'BtcvTrain': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake',
            'ann_file': 'dummy_annotations.json',
            'split': 'train'
        },
        'BtcvVal': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_test_new_snake',
            'ann_file': 'dummy_annotations.json',
            'split': 'val'
        },
        'BtcvMini': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/BTCV/btcv_png_new_snake',
            'ann_file': 'dummy_annotations.json',
            'split': 'mini'
        },
        'RaosTrain': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/RAOS/RAOS-Real/CancerImages_Set1/processed_Tr_resized',
            'ann_file': 'dummy_annotations.json',
            'split': 'train'
        },
        'RaosVal': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/RAOS/RAOS-Real/CancerImages_Set1/processed_Ts_resized',
            'ann_file': 'dummy_annotations.json',
            'split': 'val'
        },
        'RaosMini': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Datasets/RAOS/RAOS-Real/CancerImages_Set1/processed_Tr_resized',
            'ann_file': 'dummy_annotations.json',
            'split': 'mini'
        },
        'Proc1232Train': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Data_processed/1232processed',
            'ann_file': 'dummy_annotations.json',
            'split': 'train'
        },
        'Proc1232Val': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Data_processed/1232processed',
            'ann_file': 'dummy_annotations.json',
            'split': 'val'
        },
        'Proc1232Mini': {
            'id': 'sbd',
            'data_root': '/home/medteam/Zhrch/Data_processed/1232processed',
            'ann_file': 'dummy_annotations.json',
            'split': 'mini'
        },
        'VocVal': {
            'id': 'voc',
            'data_root': 'data/voc/JPEGImages',
            'ann_file': 'data/voc/annotations/voc_val_instance.json',
            'split': 'val'
        },
        'CocoTrain': {
            'id': 'coco',
            'data_root': '/home/medteam/Zhrch/COCO/train2017',
            'ann_file': '/home/medteam/Zhrch/COCO/annotations/instances_train2017.json',
            'split': 'train'
        },
        'CocoVal': {
            'id': 'coco',
            'data_root': '/home/medteam/Zhrch/COCO/val2017',
            'ann_file': '/home/medteam/Zhrch/COCO/annotations/instances_val2017.json',
            'split': 'val'
        },
        'CocoMini': {
            'id': 'coco',
            'data_root': '/home/medteam/Zhrch/COCO/train2017',
            'ann_file': '/home/medteam/Zhrch/COCO/annotations/instances_train2017.json',
            'split': 'mini'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
