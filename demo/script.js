 function getStatistics() {
	 
	$.ajax({ 
				url: "statistics.html",
				type: "GET",
				dataType: "text",
				success: function(data) {
				
				if(!data && !isStatistics)
				{
					setTimeout(function() {getStatistics();} , 3500);
					
					
				}else if(!isStatistics){
					
					
					$("#content").append(data);
					("#visible").show();
					drawGraphics
					alert(data);
					window.isStatistics=true;
				}
				
				
				
				}
	}); 
	 
	
		
	 
};
