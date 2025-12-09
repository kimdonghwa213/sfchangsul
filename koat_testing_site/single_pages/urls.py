from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = "single_pages"

urlpatterns = [
    path('about_me/', views.about_me, name='about_me'),
    #path('', views.landing_page, name='landing_page'),
    path("blog/", views.blog_list, name='blog_list'),
    path('', auth_views.LoginView.as_view(template_name='single_pages/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]
