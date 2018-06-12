 function startAnimation() {  
    
    window.r = new Raphael(document.getElementById('canvas_container'), 800, 300);  
	//var bucket = paper.path("M 250 250 l 25 -50 l -50 0 l -50 0 l 25 50 z"); 
	var circles = new Array();
	var buckets = new Array();

    for(var i = 0; i < 5; i+=1) {
     	buckets[i] = 0;
	 }

	circles[0] = 0;
	circles[1] = 1;
	circles[2] = 2;
	circles[3] = 3;
	circles[4] = 0;
	circles[5] = 2;
	circles[6] = 1;
	circles[7] = 4;
	circles[8] = 3;
	circles[9] = 4;



    for(var i = 0; i < 5; i+=1) {

		r.path("M "+(150+150*i)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#ff0d0d', 'stroke-width': 5});

	}


    set = r.set(r.circle(50+100, 50, 8).attr({stroke: "none", fill: "#ff0d0d"}),
    			r.circle(100 +100, 50, 8).attr({stroke: "none", fill: "#ffa31b"}),
    			r.circle(150+100, 50, 8).attr({stroke: "none", fill: "#d0ff17"}),
    			r.circle(200+100, 50, 8).attr({stroke: "none", fill: "#4eff23"}),
    			r.circle(250+100, 50, 8).attr({stroke: "none", fill: "#ff0d0d"}),
    			r.circle(300+100, 50, 8).attr({stroke: "none", fill: "#d0ff17"}),
    			r.circle(350+100, 50, 8).attr({stroke: "none", fill: "#ffa31b"}),
    			r.circle(400+100, 50, 8).attr({stroke: "none", fill: "#6666ff"}),
    			r.circle(450+100, 50, 8).attr({stroke: "none", fill: "#4eff23"}),
    			r.circle(500+100, 50, 8).attr({stroke: "none", fill: "#6666ff"}),
    			r.path("M "+(150+150*0)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#ff0d0d', 'stroke-width': 5}),
    			r.path("M "+(150+150*1)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#ffa31b', 'stroke-width': 5}),
    			r.path("M "+(150+150*2)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#d0ff17', 'stroke-width': 5}),
    			r.path("M "+(150+150*3)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#4eff23', 'stroke-width': 5}),
    			r.path("M "+(150+150*4)+" 250 l 25 -50 l -50 0 l -50 0 l 25 50 z").attr({stroke: '#6666ff', 'stroke-width': 5}));
 	



                        function select (id) {

                        	buckets[circles[id]] += 1;

                        	return ( 95 + 18 * buckets[circles[id]] + 150 * circles[id]);

                       };
                       
                       function select2 (id) {

                        	buckets[circles[id]] += 1;

                        	return ( 250 + 18 * buckets[circles[id]] + 50 * circles[id]);

                       };



                         function animation (id) {

                           set[id].animate({  
						    		path: set[id].path}); 
									var ex = "backOut", ey = ">";

						                    set[id].stop().animate({
						                        "100%": {cy: 242},
						                    }, 1000).animate({
						                        "100%": {cx: select(id), easing: ex},
						                    }, 1000);




                       };
                       
                      function animation2 (id) {

                           set[id].animate({  
						    		path: set[id].path}); 
									var ex = "backOut", ey = ">";

						                    set[id].stop().animate({
						                        "100%": {cy: 100},
						                    }, 1000).animate({
						                        "100%": {cx: select2(id), easing: ex},
						                    }, 1000);




                       }; 


                       animation(0);
                       setTimeout(function() {animation(1)} , 1500); 
                       setTimeout(function() {animation(2)} , 3000);
                       setTimeout(function() {animation(3)} , 4500);
                       setTimeout(function() {animation(4)} , 6000);
                       setTimeout(function() {animation(5)} , 7500); 
                       setTimeout(function() {animation(6)} , 9000); 
                       setTimeout(function() {animation(7)} , 10500); 
                       setTimeout(function() {animation(8)} , 12000);
                       setTimeout(function() {animation(9)} , 13500);
                       
                       if(window.continue == 1){
						setTimeout(function() {r.remove(); startAnimation();} , 14500);
                       }else{
                       
                       setTimeout(function() {animation2(0)} , 14500);
                       setTimeout(function() {animation2(1)} , 14500); 
                       setTimeout(function() {animation2(2)} , 14500);
                       setTimeout(function() {animation2(3)} , 14500);
                       setTimeout(function() {animation2(4)} , 14500);
                       setTimeout(function() {animation2(5)} , 14500); 
                       setTimeout(function() {animation2(6)} , 14500); 
                       setTimeout(function() {animation2(7)} , 14500); 
                       setTimeout(function() {animation2(8)} , 14500);
                       setTimeout(function() {animation2(9)} , 14500);
						}
                       
                       
                       

	
}  
