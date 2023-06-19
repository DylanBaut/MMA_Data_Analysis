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

    const submit = document.getElementById("submit-button");
    submit.addEventListener("click", validate);

    function validate(e) {

        const sigAtt1 = document.getElementById("sigAtt1")
        const sigLand1 = document.getElementById('sigLand1')
        const KD1 = document.getElementById('KD1')
        const TD1 =document.getElementById('TD1')
        const sub1 = document.getElementById('sub1')
        const ctrl1 = document.getElementById('ctrl1')
        const head1 = document.getElementById('head1')

        const sigAtt2 = document.getElementById('sigAtt2')
        const sigLand2 = document.getElementById('sigLand2')
        const KD2 = document.getElementById('KD2')
        const TD2 = document.getElementById('TD2')
        const sub2 = document.getElementById('sub2')
        const ctrl2 = document.getElementById('ctrl2')
        const head2 = document.getElementById('head2')
        
        sigAtt1.classList.remove("invalid");
        sigLand1.classList.remove("invalid");
        KD1.classList.remove("invalid");
        TD1.classList.remove("invalid");
        sub1.classList.remove("invalid");
        ctrl1.classList.remove("invalid");
        head1.classList.remove("invalid");
        sigLand2.classList.remove("invalid");
        KD2.classList.remove("invalid");
        TD2.classList.remove("invalid");
        sub2.classList.remove("invalid");
        ctrl2.classList.remove("invalid");
        head2.classList.remove("invalid");
        
        let valid = true;
        let alert = ''        
        if (parseInt(sigAtt1.value) < parseInt(sigLand1.value)) {
            sigAtt1.classList.add("invalid");
            sigLand1.classList.add("invalid");
            alert = alert + "sig strikes landed more than attempted ; "
            valid = false;
        }
        if (parseInt(sigAtt2.value) < parseInt(sigLand2.value)) {
            sigAtt2.classList.add("invalid");
            sigLand2.classList.add("invalid");
            alert = alert + "sig strikes landed more than attempted ; "
            valid = false;
        }
        if (parseInt(sigAtt1.value) < parseInt(head1.valu)) {
            sigAtt1.classList.add("invalid");
            head1.classList.add("invalid");
            alert = alert + "sig head strikes landed more than attempted; "
            valid = false;
        }
        if (parseInt(sigAtt2.value) < parseInt(head2.value)) {
            sigAtt2.classList.add("invalid");
            head2.classList.add("invalid");
            alert = alert + "sig head strikes landed more than attempted; "
            valid = false;
        }
        if (parseInt(sigLand1.value) < parseInt(head1.value)) {
            sigLand1.classList.add("invalid");
            head1.classList.add("invalid");
            alert = alert + "sig head strikes landed more than landed; "
            valid = false;
        }
        if (parseInt(sigLand2.value) < parseInt(head2.value)) {
            sigLand2.classList.add("invalid");
            head2.classList.add("invalid");
            alert = alert + "sig head strikes landed more than landed; "
            valid = false;
        }

        if (!valid) { 
            window.alert(alert)
            e.preventDefault();
        }
        return valid;
    }

    const fightScorer = document.getElementById("fightScorer");
    fightScorer.addEventListener("click", scorerRedir);  
    function scorerRedir() { 
        var base_url = window.location.origin;
        window.location = base_url + "/fightScorer"
        
    }
    const roundJudge = document.getElementById("roundJudge");
    roundJudge.addEventListener("click", roundRedir);  
    function roundRedir() { 
        var base_url = window.location.origin;
        window.location = base_url
        
    }

    $(document).ready(function () {
        $('select').selectize({
        });
    });
    var $select = $('#fight-list').selectize();
    var control = $select[0].selectize;
    control.clear();
});