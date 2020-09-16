
REQUIRES Python 3.7+
Tested version: Python 3.8.2


TODO:
Embed customer id
Convert to multi-binary classifier:
    https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification
    https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances
    https://stackoverflow.com/questions/45799474/keras-model-evaluate-vs-model-predict-accuracy-difference-in-multi-class-nlp-ta/45834857#45834857

gather probably can be replaced by boolean_mask, anywhere
Generate text file with train samples, to debug

NON SEQUENTIAL:
Score: 62793 of 151689
Ratio: 0.4139588236457489
Total time: 2.959610939025879

SEQUENTIAL:
Score: 68515 of 150192
Ratio: 0.45618275274315545
Total time: 2.914384126663208

(After pre/post processing / 1)
Score: 68951 of 151984
Ratio: 0.45367275502684495
Total time: 4.415704011917114


See https://stackoverflow.com/questions/50166420/how-to-concatenate-2-embedding-layers-with-mask-zero-true-in-keras2-0 for multiple
masked inputs (for future projects)

Requisites:
tensorflow==2.3

Installation:
python3 -m venv --system-site-packages ./venv-tf-2.3
source venv-tf-2.3/bin/activate
pip install --upgrade pip
pip install tensorflow==2.3
pip install requests # Only needed to test tf serving performance

Show exported model
saved_model_cli show --dir model/serving_model/1 --all
saved_model_cli show --dir model/serving_model/1 --tag_set serve --signature_def predict

Install serving
sudo apt install docker.io
sudo docker pull tensorflow/serving

sudo docker run -p 8501:8501 --mount type=bind,source=/home/toni/proyectos/tensorflow/basketMarket/model/serving_model,target=/models/basket -e MODEL_NAME=basket -t tensorflow/serving

curl -d '{"signature_name":"predict", "inputs": { "customer_label": "5909", "item_labels": ["21131", "221554"], "n_results": 10 } }' -X POST "http://localhost:8501/v1/models/basket:predict"

Serving time:
Total time: 0.0029418468475341797

Kill all containers:
sudo docker container kill $(sudo docker ps -q)

Troubles starting Hyper-V service in Windows: See https://github.com/docker/for-win/issues/3597#issuecomment-474949164 (worked for me)
Troubles starting Docker: Try this: cmd.exe > menu > properties > disable legacy console


Windows:
D:\kbases\subversion\basketMarket-tf>docker run -p 8501:8501 --mount type=bind,source=C:\xxxxxx\model\serving_model,target=/models/basket -e MODEL_NAME=basket -t tensorflow/serving
C:\xxxxx\venv\Scripts\activate.ps1 (or .bat)

