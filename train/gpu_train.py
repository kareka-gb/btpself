import sys
sys.path.append('..')

from data import drone_data
import tensorflow as tf
from models.edsr import edsr
from models.wdsr import wdsr_b
from models.srgan import generator, discriminator
from train import EdsrTrainer, WdsrTrainer, SrganTrainer

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print("\nPreparing Data Loaders")

train_loader = drone_data(subset='train')
train_ds = train_loader.dataset(batch_size=16)
valid_loader = drone_data(subset='valid')
valid_ds = valid_loader.dataset(batch_size=1, repeat_count=1)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nTraining EDSR Model with 30000 steps")
# EDSR training as mentioned in the paper
trainer = EdsrTrainer(model = edsr(scale=4, num_res_blocks=32),
                      checkpoint_dir=f'.ckpt/edsr-32-x4')
trainer.train(train_ds,
            valid_ds.take(10),
            steps=30000,
            evaluate_every=100,
            save_best_only=True)
trainer.restore()
os.makedirs('weights/edsr-32-x4', exist_ok=True)
trainer.model.save_weights('weights/edsr-32-x4/weights.h5')

print("\nTesting EDSR Model with 2200 validation images")
psnr = trainer.evaluate(valid_ds)
print(f'PSNR of EDSR = {psnr.numpy():5f}')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nTraining WDSR Model with 30000 steps")             
# WDSR training as mentioned in the paper
trainer = WdsrTrainer(model=wdsr_b(scale=4, num_res_blocks=32), 
                      checkpoint_dir=f'.ckpt/wdsr-b-8-x4')

trainer.train(train_ds,
              valid_ds.take(10),
              steps=30000, 
              evaluate_every=100, 
              save_best_only=True)

trainer.restore()
os.makedirs('weights/wdsr-b-32-x4', exist_ok=True)
trainer.model.save_weights('weights/wdsr-b-32-x4/weights.h5')

print("\nTesting WDSR Model with 2200 validation images")
psnr = trainer.evaluate(valid_ds)
print(f'PSNR of WDSR = {psnr.numpy():5f}')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nFinetuning EDSR Model with SRGAN discriminator for 30000 steps")
# Create EDSR generator and init with pre-trained weights
generator = edsr(scale=4, num_res_blocks=32)
generator.load_weights('weights/edsr-32-x4/weights.h5')

# Fine-tune EDSR model via SRGAN training.
gan_trainer = SrganTrainer(generator=generator, discriminator=discriminator())
gan_trainer.train(train_ds, steps=30000)
generator.save_weights('weights/edsr-32-x4/finetuned_weights.h5')

print("\nTesting EDSR + SRGAN Model with 2200 validation images")
psnr = trainer.evaluate(valid_ds)
print(f'PSNR of EDSR + SRGAN = {psnr.numpy():5f}')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nFinetuning WDSR Model with SRGAN discriminator for 30000 steps")
# Create WDSR B generator and init with pre-trained weights
generator = wdsr_b(scale=4, num_res_blocks=32)
generator.load_weights('weights/wdsr-b-32-x4/weights.h5')

# Fine-tune WDSR B  model via SRGAN training.
gan_trainer = SrganTrainer(generator=generator, discriminator=discriminator())
gan_trainer.train(train_ds, steps=300)
generator.save_weights('weights/wdsr-b-32-x4/finetuned_weights.h5')

print("\nTesting WDSR + SRGAN Model with 2200 validation images")
psnr = trainer.evaluate(valid_ds)
print(f'PSNR of WDSR + SRGAN = {psnr.numpy():5f}')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
