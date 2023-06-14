$(document).ready(function () {
    $("p.desc").hide();
    $("#modelJudge").show();

    $("input[name$='modelSel']").click(function() {
        var test = $(this).val();
        $("p.desc").hide();
        $("#model" + test).show();
    });

    var perc1 = $("#prog1").text()
    var lim1 = perc1.substring(0, perc1.length - 1);
    if (lim1 != undefined) {
        let width = 1;
        let id = setInterval(frame, 100);

        function frame() {
            if (width >= lim1) {
                clearInterval(id);
            } else {
                width++;
                $("#progress1").css({ "width": width + '%' });
            }
        }
    }
    
    var perc2 = $("#prog2").text()
    var lim2 = perc2.substring(0, perc2.length - 1);
    if (lim2 != undefined) { 
        let width2 = 1;
        let id2 = setInterval(frame2, 100);

        function frame2() {
            if (width2 >= lim2) {
                clearInterval(id2);
            } else {
                width2++;
                $("#progress2").css({ "width": width2 + '%' });
            }
        }
    }


    var perc3 = $("#prog3").text()
    var lim3 = perc3.substring(0, perc3.length - 1);
    if (lim3 != undefined) { 
        let width3 = 1;
        let id3 = setInterval(frame3, 100);
        function frame3() {
            if (width3 >= lim3) {
                clearInterval(id3);
            } else {
                width3++;
                $("#progress3").css({ "width": width3 + '%' });
            }
        }
    }
    
    var perc4 = $("#prog4").text()
    var lim4 = perc4.substring(0, perc4.length - 1);
    if (lim4 != undefined) { 
        let width4 = 1;
        let id4 = setInterval(frame4, 100);

        function frame4() {
            if (width4 >= lim4) {
                clearInterval(id4);
            } else {
                width4++;
                $("#progress4").css({ "width": width4 + '%' });
            }
        }
    }

    var perc5 = $("#prog5").text()
    var lim5 = perc5.substring(0, perc5.length - 1);
    if (lim5 != undefined) { 
        let width5 = 1;
        let id5 = setInterval(frame5, 100);

        function frame5() {
            if (width5 >= lim5) {
                clearInterval(id5);
            } else {
                width5++;
                $("#progress5").css({ "width": width5 + '%' });
            }
        }
    }
});