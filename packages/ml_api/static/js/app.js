$(document).ready(function(){
    $(function(){
       $("loading").hide();
    });

    $(function(){
        var $select_0 = $(".1-25");
        var $select_1 = $(".0-100");
        var $select_2 = $(".1-43");

        for (i=1;i<=25;i++){
            $select_0.append($('<option></option>').val('type ' + i).html('type ' + i))
        }

        for (j=0;j<=100;j++){
            $select_1.append($('<option></option>').val('type ' + j).html('type ' + j))
        }

        for (i=1;i<=43;i++){
            $select_2.append($('<option></option>').val('type ' + i).html('type ' + i))
        }
    });
});


