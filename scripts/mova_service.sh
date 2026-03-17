sglang generate \
  --model-path '/inspire/qb-ilm/project/control-technology/public/models/MOVA-720p' \
  --prompt "A man in a blue blazer and glasses speaks in a formal indoor setting, \
  framed by wooden furniture and a filled bookshelf. \
  Quiet room acoustics underscore his measured tone as he delivers his remarks. \
  At one point, he says, \"I would also believe that this advance in AI recently wasn’t unexpected.\"" \
  --image-path "./assets/single_person.jpg" \
  --adjust-frames false \
  --num-gpus 8 \
  --ring-degree 2 \
  --ulysses-degree 4 \
  --num-frames 193 \
  --fps 24 \
  --seed 67 \
  --num-inference-steps 25 \
  --enable-torch-compile \
  --save-output