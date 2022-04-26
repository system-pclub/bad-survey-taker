  /**
   * decide attention check position (from 0 to numAttentionCheckPos -1)
   */

function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
  }
  
  let mouseMoved = false;

  let responseId = "${e://Field/ResponseID}"
  let qualtricsIp = "${loc://IPAddress}" 
  
  Qualtrics.SurveyEngine.addOnload(function()
  {
      /*Place your JavaScript here to run when the page loads*/
      // set position of attention check
      let num = parseInt("${e://Field/numAttentionCheckPos}")
      let pos = getRandomInt(0, num)
      Qualtrics.SurveyEngine.setEmbeddedData( 'attentionCheckPos', pos.toString() )
      // set show explanation or solution
  });
  
  Qualtrics.SurveyEngine.addOnReady(function()
  {
  });
  
  Qualtrics.SurveyEngine.addOnUnload(function()
  {
  });