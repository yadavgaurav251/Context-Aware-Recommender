
fetch("http://api.quotable.io/random")
.then(data=>data.json())
.then(jokeData=>{
    const jokeText=jokeData.content;
    const jokeElement=document.getElementById('jokeElement')

    jokeElement.innerHTML=jokeText
})

