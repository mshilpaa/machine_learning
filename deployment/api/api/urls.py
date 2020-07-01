from django.contrib import admin
from django.urls import path
from predictor.views import *
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('knn_form/', knn_get),
    path('cnn_form/', cnn),
    path('cnn_result/', cnn),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)