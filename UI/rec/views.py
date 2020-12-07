from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    if request.method=='POST':
        movies_id = [[862, 96], [8884, 95], [284, 93], [1907, 98], [1285, 95], [867, 87], [337, 95], [10527, 67], [1998, 78], [580, 95],  [10527, 95], [874, 95]]
        choice = request.POST['choice']
        movie = request.POST['movie']
        userId = request.POST['user_Id']
        if choice=="Yes":
            numberOfPeople = request.POST['number_of_people'] 
            device = request.POST['device-type']
            mood = request.POST['mood']
            family = request.POST.getlist('family[]')
        return render(request,'rec/index.html', {'display_rec': 'inline-block','movies_id':movies_id})

    return render(request,'rec/index.html',{'yes':False, 'no':False, 'movie':"Hello", 'number':0, 'display_rec': 'none'})


def movie(request, id):
    return render(request,'rec/movie.html', {'id':id})
    