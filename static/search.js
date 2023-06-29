$(document).ready(function () {
    
    var perc1 = $("#path1").text()
    var rd = perc1.split(" ")
    perc1 = rd[2]
    var lim1 = parseFloat(perc1.substring(0, perc1.length - 1));

    var perc2 = $("#path2").text()
    var rd2 = perc2.split(" ")
    perc2 = rd2[2]
    var lim2 = parseFloat(perc2.substring(0, perc2.length - 1));

    var perc3 = $("#path3").text()
    var rd3 = perc3.split(" ")
    perc3 = rd3[2]
    var lim3 = parseFloat(perc3.substring(0, perc3.length - 1));

    var perc4 = $("#path4").text()
    var rd4 = perc4.split(" ")
    perc4 = rd4[2]
    var lim4 = parseFloat(perc4.substring(0, perc4.length - 1));

    var perc5 = $("#path5").text()
    var rd5 = perc5.split(" ")
    perc5 = rd5[2]
    var lim5 = parseFloat(perc5.substring(0, perc5.length - 1));
    var tot = lim1 + lim2 + lim3 + lim4 + lim5
    
    lim1 = (lim1 / tot)*100
    lim2 = (lim2 / tot)*100   
    lim3 = (lim3 / tot)*100   
    lim4 = (lim4 / tot)*100
    lim5 = (lim5 / tot)*100

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

    console.log(lim1+"1")
    console.log(lim2+"2")
    console.log(lim3)
    console.log(lim4)
    console.log(lim5)


    function csvToArray(str, delimiter = ",") {
        var $select = $('#fight-list').selectize({
            maxOptions: 500,
            maxItems: 500,
        })

        var selectizeO = $select[0].selectize
        
        var fightList = document.getElementById('fight-list')
        let array = str.split('\n')
        let i = 0
    
        array.slice(1, -1).forEach(element => {
            let bout = element.split(',')
            let label = bout[0] + " vs. " + bout[1]
            selectizeO.addOption({ value: String(i), text: String(label)});
            i++
        });
    }
    

    fetch('static/data/scoresData.csv')
        .then(response => response.text())
        .then(text => csvToArray(text))


});