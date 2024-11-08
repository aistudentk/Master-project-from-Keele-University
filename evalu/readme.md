##  evaluating

To reproduce the evaluation results in the paper for num_points 10, weight_exponent 1, you can use the following command:
```bash
python eval.py --num_points 10 --weight_exponent 1 --cam_method gradcam --model_weight Imagenet_pretrained
```
## to get a finetuned model on DIC_crack_dataset

```
cd class2seg

python train.py --pretrained
```