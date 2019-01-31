var gatys_active = false;
var gan_active = false;

//-----------------------------------------------------------------------------
//  Start generating image using gatys method
function gatys_button(){
    gatys_active = true;
    var source = $('#source_select').val()
    var params = '?source=' + source;
    $('#gatys_button').hide()
    $('#gatys_progress_container').show();
    $.get('start_gatys' + params, function( data ) {
        gatys_active = false;
        $('#gatys_progress_container').hide();
        $('#gatys_button').show()
    });
}

//-----------------------------------------------------------------------------
//  Start generating image using gan method
function gan_button(){
    gan_active = true;
    var source = $('#source_select').val()
    var params = '?source=' + source;
    $('#gan_button').hide()
    $('#gan_progress_container').show();
    $.get('start_gan' + params, function( data ) {
        gan_active = false;
        $('#gan_progress_container').hide();
        $('#gan_button').show()
    });
}

//-----------------------------------------------------------------------------
//  Update progress bar for gatys/gan
function update_progress(subject){
    $.get('progress?subject=' + subject, function( data ) {
        percent = parseInt(data);
        $('#' + subject + '_progress').css('width', percent + '%');
    });
}

//-----------------------------------------------------------------------------
//  Refresh image for gatys/gan 
function refresh_image(subject){
    var source = $('#source_select').val()
    d = new Date();
    $("#"+ subject +"_image").attr("src", 'image?target=temp/'+ subject +'.jpg&alt=textures/cropped/' + source + '.jpg&time=' + d.getTime());

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
});

//-----------------------------------------------------------------------------
//  Loop
setInterval(function(){
    if(gatys_active){
        update_progress('gatys');
        refresh_image('gatys');
    }
    if(gan_active){
        update_progress('gan');
        refresh_image('gan');
    }
}, 400)