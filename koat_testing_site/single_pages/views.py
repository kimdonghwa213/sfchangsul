from django.contrib.auth.decorators import login_required
from django.shortcuts import render

# Create your views here.
@login_required
def about_me(request):
    return render(
        request,
        'single_pages/about_me.html',
        {'title': 'About Me'}
    )

@login_required
def blog_list(request):
    post_list = [
        {
            'title': 'First Post',
            'content': 'This is my first Post',
        },
        {
            'title': 'Second Post',
            'content': 'This is my second Post',
        },
    ]
    return render(request, 'single_pages/blog.html', {'title' : 'Blog List', 'posts': post_list})