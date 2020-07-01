from django.shortcuts import render
from .apps import PredictorConfig
from django.http import HttpResponse
from .inference import Predict
import torchvision
from django.conf import settings
from PIL import Image

def knn_get(request):
    if request.method == 'POST':
        # get sound from request
        
        addr_info = [  float(request.POST.get('no_of_out_transactions')),
            float(request.POST.get( 'tot_ether_sent')),
            float(request.POST.get('no_of_in_transactions')), 
            float(request.POST.get( 'tot_ether_recieved')),
            float(request.POST.get( 'monthly_out_txn')),
            float(request.POST.get('monthly_in_txn')),
            float(request.POST.get( 'active_months')),
            float(request.POST.get( 'eth_balance')),
            float(request.POST.get( 'time_bw_out_txn')),
            float(request.POST.get('time_bw_in_txn')),
            float(request.POST.get( 'tot_token_value_recieved')),
            float(request.POST.get( 'tot_token_value_sent')),
            float(request.POST.get('monthly_ether_sent')),
            float(request.POST.get( 'monthly_ether_recieved'))]


        # normalize 
        # scalar = PredictorConfig.sc_x.transform([addr_info])
        # predict based on vector
        # print('----------------------------------',addr_info)
        prediction = PredictorConfig.knn.predict([addr_info])
        # build response
        # print('------------------',prediction[0])
        if str(prediction[0]) == 0:
            response = 'Not Malicious'
        else:
            response = 'Malicious'
        # return response
        return HttpResponse(response)

    return render(request,'knn_form.html')

def cnn(request):
    if request.method == 'POST':
        print('----------------------',request.FILES)
        fgbg = request.FILES['fgbg']
        bg = request.FILES['bg']

        # save the uploaded files to media/images
        
        # img_extension = os.path.splitext(fgbg.name)[1]
        fgbg_save_path = settings.MEDIA_ROOT+'images/'+ fgbg.name
        bg_save_path = settings.MEDIA_ROOT+'images/'+ bg.name

        
        with open(fgbg_save_path, 'wb+') as f:
            for chunk in fgbg.chunks():
                f.write(chunk)

        with open(bg_save_path, 'wb+') as f:
            for chunk in bg.chunks():
                f.write(chunk)

        fgbg_path,bg_path,mask_path,depth_path = Predict(fgbg_file=fgbg.name, bg_file=bg.name)
        print('paths-----------',mask_path,depth_path)
        return render(request,'cnn_result.html',{'fgbg_path':fgbg_path,'bg_path':bg_path,'mask_path':mask_path,'depth_path':depth_path}) 

    return render(request,'cnn_form.html')
