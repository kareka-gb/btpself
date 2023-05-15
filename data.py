import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

"""
 Data size : 12
"""

class drone_data:
    def __init__(self, subset='train', images_dir='/home/kareka/Academics/btpself/data') -> None:
        if subset == 'train':
            self.image_ids = range(1, 10001)
        elif subset == 'valid':
            self.image_ids = range(10001, 12201)
        elif subset == 'test':
            self.image_ids = range(12180, 12190)
        else:
            raise ValueError("subset must be 'train' or 'valid'")
        
        self.images_dir = images_dir
    
    def __len__(self):
        return len(self.image_ids)
    
    def dataset(self, batch_size=16, repeat_count=None):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    def hr_dataset(self):
        ds = self._images_dataset(self._hr_image_files())
        return ds

    def lr_dataset(self):
        ds = self._images_dataset(self._lr_image_files())
        return ds

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, f'{image_id}_04.png') for image_id in self.image_ids]

    def _hr_images_dir(self):
        return os.path.join(self.images_dir, f'drone_rgb_hr')

    def _lr_images_dir(self):
        return os.path.join(self.images_dir, f'drone_rgb_lr')
    
    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds
