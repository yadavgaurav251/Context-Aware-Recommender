from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    if request.method=='POST':
        choice = request.POST['choice']
        movie = request.POST['movie']
        number = request.POST['num_of_movie']
        device = request.POST['device-size']
        mood = request.POST['mood']
        movies_id = [[862, 96], [8884, 95], [284, 93], [1907, 98], [1285, 95], [867, 87], [337, 95], [10527, 67], [1998, 78], [580, 95],  [10527, 95], [874, 95]]
        return render(request,'rec/index.html', {'display_rec': 'inline-block','movies_id':movies_id})

    return render(request,'rec/index.html',{'yes':False, 'no':False, 'movie':"Hello", 'number':0, 'display_rec': 'none'})


def movie(request, id):
    return render(request,'rec/movie.html', {'id':id})
    