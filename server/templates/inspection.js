function is_ipv4(d) {
    return webrtc_regex_v4.test(d)
  }
  function is_ipv6(d) {
    return webrtc_regex_v6.test(d)
  }
  var simpleIPRegex = /([0-9]{1,3}(\.[0-9]{1,3}){3}|[a-f0-9]{1,4}(:[a-f0-9]{1,4}){7})/g;
  let webrtc_regex_v4 = /((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])/
  , webrtc_regex_v6 = /((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))/
  , ip_regex_array = [webrtc_regex_v6, webrtc_regex_v4];
  function peer(d) {
    var e, t = window.RTCPeerConnection || window.mozRTCPeerConnection || window.webkitRTCPeerConnection;
    function n() {
        (e = new t({
            iceServers: [{
                urls: "stun:stun.l.google.com:19302"
            }]
        })).onicecandidate = (d=>f(d)),
        e.createDataChannel("fake_data_channel")
    }
    function a() {
        return e.createOffer().then(d=>e.setLocalDescription(d))
    }
    function f(e) {
        e && e.candidate && e.candidate.candidate && d(e && e.candidate && e.candidate.candidate)
    }
    return {
        start: function() {
            n(),
            a()
        },
        stop: function() {
            if (e)
                try {
                    e.close()
                } finally {
                    e.onicecandidate = (()=>{}
                    ),
                    e = null
                }
        },
        createConnection: n,
        createStunRequest: a,
        handleCandidates: f
    }
  }
  function publicIPs(d) {
    if (d && d < 100)
        throw new Error("Custom timeout cannot be under 100 milliseconds.");
    var e = []
      , t = peer(function(d) {
        var t = [];
        for (let e of ip_regex_array) {
            let n = []
              , a = e.exec(d);
            if (a) {
                for (let d = 0; d < a.length; d++)
                    (is_ipv4(a[d]) || is_ipv6(a[d])) && n.push(a[d]);
                t.push(n)
            }
        }
        !function(d) {
            e.includes(d) || e.push(n(d.flat(1 / 0)))
        }(t.flat(1 / 0))
    });
    function n(d) {
        return [...new Set(d)]
    }
    return new Promise(function(a, f) {
        t.start(),
        setTimeout(()=>{
            e && e !== [] ? a(n(e.flat(1 / 0))) : f("No IP addresses were found."),
            t.stop()
        }
        , d || 500)
    }
    )
  }
  function getIPTypes(d) {
    return new Promise(function(e, t) {
        let n = [];
        publicIPs(d).then(d=>{
            d.forEach(d=>{
                d.match(/^(192\.168\.|169\.254\.|10\.|172\.(1[6-9]|2\d|3[01]))/) ? n.push({
                    ip: d,
                    type: "private",
                    IPv4: !0
                }) : d.match(/((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))/) ? n.push({
                    ip: d,
                    type: "IPv6",
                    IPv4: !1
                }) : n.push({
                    ip: d,
                    type: "public",
                    IPv4: !0
                })
            }
            ),
            e(n)
        }
        ).catch(t)
    }
    )
  }
  function getIPv4(d) {
    return getIPTypes(d).then(d=>{
        const e = d.filter(d=>d.IPv4);
        for (let d = 0; d < e.length; d++)
            e[d] = e[d].ip;
        return e || ""
    }
    )
  }
  function getIPv6(d) {
    return getIPTypes(d).then(d=>{
        const e = d.filter(d=>"IPv6" === d.type);
        for (let d = 0; d < e.length; d++)
            e[d] = e[d].ip;
        return e ? e.ip : ""
    }
    )
  }
  function getIPs(d) {
    return Object.assign(publicIPs(d), {
        types: getIPTypes,
        public: publicIPs,
        IPv4: getIPv4,
        IPv6: getIPv6
    })
  }
  
  
  var docCookies = {
  getItem: function (sKey) {
    return decodeURIComponent(document.cookie.replace(new RegExp("(?:(?:^|.*;)\\s*" + encodeURIComponent(sKey).replace(/[-.+*]/g, "\\$&") + "\\s*\\=\\s*([^;]*).*$)|^.*$"), "$1")) || null;
  },
  setItem: function (sKey, sValue, vEnd, sPath, sDomain, bSecure) {
    if (!sKey || /^(?:expires|max\-age|path|domain|secure)$/i.test(sKey)) { return false; }
    var sExpires = "";
    if (vEnd) {
      switch (vEnd.constructor) {
        case Number:
          sExpires = vEnd === Infinity ? "; expires=Fri, 31 Dec 9999 23:59:59 GMT" : "; max-age=" + vEnd;
          break;
        case String:
          sExpires = "; expires=" + vEnd;
          break;
        case Date:
          sExpires = "; expires=" + vEnd.toUTCString();
          break;
      }
    }
    document.cookie = encodeURIComponent(sKey) + "=" + encodeURIComponent(sValue) + sExpires + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "") + (bSecure ? "; secure" : "");
    return true;
  },
  removeItem: function (sKey, sPath, sDomain) {
    if (!sKey || !this.hasItem(sKey)) { return false; }
    document.cookie = encodeURIComponent(sKey) + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT" + ( sDomain ? "; domain=" + sDomain : "") + ( sPath ? "; path=" + sPath : "");
    return true;
  },
  hasItem: function (sKey) {
    return (new RegExp("(?:^|;\\s*)" + encodeURIComponent(sKey).replace(/[-.+*]/g, "\\$&") + "\\s*\\=")).test(document.cookie);
  },
  keys: /* optional method: you can safely remove it! */ function () {
    var aKeys = document.cookie.replace(/((?:^|\s*;)[^\=]+)(?=;|$)|^\s*|\s*(?:\=[^;]*)?(?:\1|$)/g, "").split(/\s*(?:\=[^;]*)?;\s*/);
    for (var nIdx = 0; nIdx < aKeys.length; nIdx++) { aKeys[nIdx] = decodeURIComponent(aKeys[nIdx]); }
    return aKeys;
  }
  };
  
    
  function initFingerprintJS() {
    // Initialize an agent at application startup.
    const fpPromise = FingerprintJS.load()
    // Get the visitor identifier when you need it.
    return fpPromise
      .then(fp => fp.get())
      .then(result => {
        // This is the visitor identifier:
        const visitorId = result.visitorId
        return result
      })
  }
  
  // "${loc://IPAddress}" "${e://Field/ResponseID}"
  function sendIP(ip, responseId) {
    let url = "//rec.nemo-arxiv.club/posts/qualtrics"
    let data = {
      ip: ip,
      response_id: responseId
    }
    fetch(
          url, {
          body: JSON.stringify(data),
          headers: {
            'content-type': 'application/json',
          },
          mode: 'cors',
          method: 'POST'
        })
          .then(response => {
            response.text().then(text => {
            })
          })
  }
  
  // below are in qualtrics question
  function appendEmbeddedData(field, data) {
    let old = Qualtrics.SurveyEngine.getEmbeddedData(field)
    let qid = this.questionId
    let new_data = old + "\n" + qid + ": " + data
    Qualtrics.SurveyEngine.setEmbeddedData(field, new_data)
  }
  
  // "${e://Field/ResponseID}"
  function recordCookies(responseId) {
    let name = "x_survey_accessed"
    let cookie = docCookies.getItem(name)
    if (cookie) {
    appendEmbeddedData("cookie", cookie)
    } else {
      docCookies.setItem(name, responseId, Infinity)
    }
  }
  
  function recordFp() {
    let fpPromise = initFingerprintJS()
    fpPromise.then(result => {
      let hash = result.visitorId
    appendEmbeddedData("browserFingerprint", hash)
    })
  }
  
  function recordWebRtcIp() {
      let webrtcPromise = getIPs(2000)
    webrtcPromise.then(webrtcData => {
        let webrtcStr = webrtcData.join(", ")
    appendEmbeddedData("webrtcIP", webrtcStr)
    })
  }
  
             
  function recordTimezone() {
      let client_timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
    appendEmbeddedData("clientOSTimezone", client_timezone)
  }
  
  function recordData() {
      recordFp()
    recordWebRtcIp()
    recordTimezone()
    recordCookies()
  }
  