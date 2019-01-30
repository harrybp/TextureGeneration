var gatys_active = false;
var gatys_target = null;

//-----------------------------------------------------------------------------
//  Display the selected source image
function show_source_image(id){
    var source = $('#source_' + id).val()
    d = new Date();
    $("#img_" + id).attr("src", 'image?target=textures/cropped/'+ source + '&time=' +d.getTime());
}

//-----------------------------------------------------------------------------
//  Start generating image using gatys method
function gatys_button(){
    console.log('gatys');
    gatys_active = true;

    //Get parameters
    var source = $('#source_gatys').val()
    var lr = $('#lr_gatys').val()
    var iterations = $('#iter_gatys').val()
    var tile = $('#tile_check').prop('checked')
    var params = '?source=textures/' + source + '&target=gatys' + '&iterations=' + iterations + '&learning_rate=' + lr + '&tile=' + tile;

    //Show progress bar
    $('#gatys-progress-container').show();
    $('#gatys-progress').attr('aria-valuemax', iterations);
    $('#gatys-progress').attr('aria-valuenow', 0);

    $.get('start_gatys' + params, function( data ) {
        gatys_active = false;
        $('#gatys-progress-container').hide();
        if(tile){
            console.log('tiled')
            window.open('tile_image/gatys', '_blank');
        }
        });
}

//-----------------------------------------------------------------------------
//  Generate an image using selected GAN
function gan_demo_button(){
    var iters =  $('#source_generator').val()
    var name = $("#source_generator option:selected").text();
    $.get('demo_GAN?name=' + name + '&iters=' + iters, function( data ) {
        d = new Date();
        $('#img-gan-demo').attr('src', 'image?target=temp/GAN_demo.jpg&time=' +d.getTime());
        });
}

//-----------------------------------------------------------------------------
//  Populate the source image dropdown list
function get_source_images(){
    $.get('get_source_images', function( data ) {
        for(var i = 0; i < data.length; i++){
            var o = new Option(data[i], data[i]);
            $("#source_gatys").append(o);
        }  
        show_source_image('gatys');  
    });
}

//-----------------------------------------------------------------------------
//Populate the generators dropdown list
function get_generators(){
    $.get('get_generators', function( data ) {
        for(var i = 0; i < data.length; i++){
            var o = new Option(data[i][0], data[i][1]);
            $("#source_generator").append(o);
        }  
        gan_demo_button();
    });
}

//-----------------------------------------------------------------------------
//  Update the progress bar for when generating image using gatys method
function update_progress_gatys(){
    $.get('progress', function( data ) {
        console.log('progress: ' + data);
        percent = parseInt(data);
        $('#gatys-progress').attr('aria-valuenow', parseInt(data)).css('width', percent + '%');
    });
}


//-----------------------------------------------------------------------------
//  Update the image displayed while generating using gatys method in progress
function refresh_gatys(){
    var source = $('#source_gatys').val()
    d = new Date();
    console.log('refresh: ')
    $("#img_gatys").attr("src", 'image?target=temp/gatys/gatys.jpg&alt=textures/' + source + '&time=' +d.getTime());
}

//-----------------------------------------------------------------------------
//  Loop
var counter = 0;
setInterval(function(){
    counter++;
    if(gatys_active){
        update_progress_gatys();
        refresh_gatys();
    }
}, 400)

//-----------------------------------------------------------------------------
//  Initialise
$(document).ready(function() {
    $('#gatys-progress-container').hide();
    get_source_images();
    get_generators();
});