var gatys_active = false;
var gan_active = false;

//-----------------------------------------------------------------------------
//  Start generating image using gatys method
function gatys_button(){
    gatys_active = true;
    //Get parameters
    var source = $('#source_select').val()
    var params = '?source=' + source + '&target=gatys' + '&iterations=300&learning_rate=0.8&tile=False';
    //Show progress bar
    $('#gatys_button').hide()
    $('#gatys_progress_container').show();
    $.get('start_gatys' + params, function( data ) {
        gatys_active = false;
        $('#gatys_progress_container').hide();
        $('#gatys_button').show()
        });
}

//-----------------------------------------------------------------------------
//  Start generating image using gatys method
function gan_button(){
    gan_active = true;
    //Get parameters
    var source = $('#source_select').val()
    var params = '?source=' + source;
    //Show progress bar
    $('#gan_button').hide()
    $('#gan_progress_container').show();
    $.get('start_gan' + params, function( data ) {
        gan_active = false;
        $('#gan_progress_container').hide();
        $('#gan_button').show()
        });
}

//-----------------------------------------------------------------------------
//  Update the progress bar for when generating image using gatys method
function update_progress_gan(){
    $.get('progress_gan', function( data ) {
        percent = parseInt(data);
        $('#gan_progress').css('width', percent + '%');
    });
}

//-----------------------------------------------------------------------------
//  Update the image displayed while generating using gatys method in progress
function refresh_gan(){
    var source = $('#source_select').val()
    d = new Date();
    $("#gan_image").attr("src", 'image?target=temp/gan/gan.jpg&alt=textures/' + source + '&time=' +d.getTime());
}

//-----------------------------------------------------------------------------
//  Update the progress bar for when generating image using gatys method
function update_progress_gatys(){
    $.get('progress', function( data ) {
        percent = parseInt(data);
        $('#gatys_progress').css('width', percent + '%');
    });
}

//-----------------------------------------------------------------------------
//  Update the image displayed while generating using gatys method in progress
function refresh_gatys(){
    var source = $('#source_select').val()
    d = new Date();
    $("#gatys_image").attr("src", 'image?target=temp/gatys/gatys.jpg&alt=textures/' + source + '&time=' +d.getTime());
}

//-----------------------------------------------------------------------------
//  Display the selected source image
function show_source_image(){
    var source = $('#source_select').val()
    d = new Date();
    $("#source_image").attr("src", 'image?target=textures/cropped/'+ source + '.jpg&time=' +d.getTime());
}

//-----------------------------------------------------------------------------
//  Populate the source image dropdown list
function get_source_images(){
    $.get('get_source_images', function( data ) {
        console.log(data)
        for(var i = 0; i < data.length; i++){
            var o = new Option(data[i].substring(0,data[i].length-4), data[i].substring(0,data[i].length-4));
            $("#source_select").append(o);
        }  
        show_source_image();  
    });
}

//-----------------------------------------------------------------------------
//  Initialise
$(document).ready(function() {
    $('#gatys_progress_container').hide();
    $('#gan_progress_container').hide();
    get_source_images();
    //get_generators();
});

//-----------------------------------------------------------------------------
//  Loop
var counter = 0;
setInterval(function(){
    counter++;
    if(gatys_active){
        update_progress_gatys();
        refresh_gatys();
    }
    if(gan_active){
        update_progress_gan();
        refresh_gan();
    }
}, 400)