Raphael.fn.pieChart = function (cx, cy, r, values, labels, stroke) {
    var paper = this,
        rad = Math.PI / 180,
        chart = this.set();
    function sector(cx, cy, r, startAngle, endAngle, params) {
        var x1 = cx + r * Math.cos(-startAngle * rad),
            x2 = cx + r * Math.cos(-endAngle * rad),
            y1 = cy + r * Math.sin(-startAngle * rad),
            y2 = cy + r * Math.sin(-endAngle * rad);
        return paper.path(["M", cx, cy, "L", x1, y1, "A", r, r, 0, +(endAngle - startAngle > 180), 0, x2, y2, "z"]).attr(params);
    }
    var angle = 0,
        total = 0,
        start = 0,
        process = function (j) {
            var value = values[j],
                angleplus = 360 * value / total,
                popangle = angle + (angleplus / 2),
                color = Raphael.hsb(start, .75, 1),
                ms = 500,
                delta = 30,
                bcolor = Raphael.hsb(start, 1, 1),
                p = sector(cx, cy, r, angle, angle + angleplus, {fill: "90-" + bcolor + "-" + color, stroke: stroke, "stroke-width": 3}),
                txt = paper.text(cx + (r + delta + 55) * Math.cos(-popangle * rad), cy + (r + delta + 25) * Math.sin(-popangle * rad), labels[j]).attr({fill: bcolor, stroke: "none", opacity: 0, "font-size": 20});
            p.mouseover(function () {
                p.stop().animate({transform: "s1.1 1.1 " + cx + " " + cy}, ms, "elastic");
                txt.stop().animate({opacity: 1}, ms, "elastic");
            }).mouseout(function () {
                p.stop().animate({transform: ""}, ms, "elastic");
                txt.stop().animate({opacity: 0}, ms);
            });
            angle += angleplus;
            chart.push(p);
            chart.push(txt);
            start += .1;
        };
    for (var i = 0, ii = values.length; i < ii; i++) {
        total += values[i];
    }
    for (i = 0; i < ii; i++) {
        process(i);
    }
    return chart;
};

function drawGraphics() {

    var values = [],
        labels = [];
    $(".rows").each(function () {
        values.push(parseInt($("td", this).text(), 10));
        labels.push($("th", this).text());
    });
    $("table").hide();
    Raphael("holder", 500, 500).pieChart(250, 250, 100, values, labels, "#fff");
};

function getStatistics() {
	 
	$.ajax({ 
				url: "auxfiles/statistics.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
				
				if(!data && !isStatistics)
				{
					//setTimeout(function() {getStatistics();} , 3500);
					
					
				}else if(!isStatistics){
					
					window.isStatistics=true;
					$("#table").append(data);
					$("#visible").show();
					drawGraphics();
					$(".canvas_container").hide();
					//getGPU();
					getNumbers();
					getBench();
					getExecution();
					window.continue = 0;
				}
				
				
				
				}
	}); 
	 
	
		
	 
};

function getGPU() {
	 $.ajax({ 
				url: "auxfiles/gpuinfo.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					
				}
	});
};

function getNumbers() {
	 $.ajax({ 
				url: "auxfiles/numbers.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					$("#sorted").append(data);
				}
	});	 
};

function getBench() {
	 $.ajax({ 
				url: "auxfiles/bench.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					
					$("#bench").append(data);
					$("#data").hide();
					loadAnalitics();
					
				}
				
	});	 
	
	 $.ajax({ 
				url: "auxfiles/benchdata.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					
					$("#infob").append(data);
					//$("#data").hide();
					//loadAnalitics();
					
				}
	});	 
	
};

function getExecution() {
	 $.ajax({ 
				url: "auxfiles/execution.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					
					$("#exec").append(data);
					
					
				}
				
	});	 
	
	$.ajax({ 
				url: "auxfiles/gpudetails.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					
					$("#exec_info").append(data);
					
					
				}
				
	});	
	
};

function startExecution() {
	 $.ajax({ 
				url: "auxfiles/stat.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
					if(data)
					{
						
						$("#scat").css('color', "#555555");
						$("#histo").css('color', "#555555");
						$("#globol").css('color', "#555555");
						$("#sort").css('color', "#555555");
						$("#graph_sub").text("Executing...");

						if(data==1){
							
							$("#scat").css('color',"#ff0d0d");
							setTimeout(function() {startExecution();} , 1000);
							return;
						}
						if(data==2){
							
							$("#histo").css('color',"#ffa31b");
							setTimeout(function() {startExecution();} , 1000);
							return;
						}
						if(data==3){
							
							$("#globol").css('color',"#d0ff17");
							setTimeout(function() {startExecution();} , 1000);
							return;
						}
						if(data==4){
							
							$("#sort").css('color',"#4eff23");
							setTimeout(function() {startExecution();} , 1000);
							return;
						}
						
						
						
						if(data==5){
							$("#graph_sub").text("Step execution times");
							window.isStatistics = false;
							//$("#visible").hide();
							getStatistics();
							return;
						}
					}
					
					
				}
				
	});	 
	
};







