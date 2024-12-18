function getxmlhttp() {
	var xmlhttp;
	if(window.XMLHttpRequest) {
		// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();
	} else if(window.ActiveXObject) {
		// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	} else {
		alert("Your browser does not support XMLHTTP!");
	}
	return xmlhttp;
}

function toggle(id) {
	/* toggle element with id to be hidden or not. */
	var el = document.getElementById(id);
	if(el.style.display == 'none')
		el.style.display = 'block';
	else
		el.style.display = 'none';
}

function showhide(id1, id2, id3, n) {
	/* show id1, hide id2; hide dd0 .. dd<n>, but show id3*/
	document.getElementById(id1).style.display = 'block';
	document.getElementById(id2).style.display = 'none';
	for (var i=0; i < n; i++) {
		document.getElementById('dd' + i).style.display = 'none';
	}
	document.getElementById(id3).style.display = 'block';
}

function togglelink(id) {
	/* toggle element with id to be hidden or not, and also toggle
	 * link with id 'link'+id to start with 'show' or 'hide'. */
	var el = document.getElementById(id);
	var link = document.getElementById('link' + id);
	if(el.style.display == 'none') {
		el.style.display = 'block';
		link.innerHTML = 'hide' + link.innerHTML.substring(4);
	} else {
		el.style.display = 'none';
		link.innerHTML = 'show' + link.innerHTML.substring(4);
	}
}

function toggletextbox() {
	/* toggle a textbox to be single or multi line. */
	var state = document.queryform.textarea;
	var cur = document.queryform.query;
	var next = document.queryform.notquery;
	var link = document.getElementById('textboxlink');
	cur.name = 'notquery';
	cur.disabled = true;
	cur.style.display = 'none';
	next.name = 'query';
	next.disabled = false;
	next.style.display = 'block';
	if(state.disabled) {
		state.disabled = false;
		next.innerHTML = cur.value;
		link.innerHTML = 'smaller';
	} else {
		state.disabled = true;
		next.value = cur.value;
		link.innerHTML = 'larger';
	}
}

function show(id, name) {
	/* show element with 'id' and enable (un-disable) all form elements with 'name'. */
	var el = document.getElementById(id);
	if(el.style.visibility != 'visible')
		el.style.visibility = 'visible';
	if(name != '') {
		var elems = document.getElementsByName(name);
		for (var n = 0; n < elems.length; n++)
			elems[n].disabled = false;
	}
}

function hide(id, name) {
	/* hide element with 'id' and disable all form elements with 'name'. */
	var el = document.getElementById(id);
	if(el.style.visibility != 'hidden')
		el.style.visibility = 'hidden';
	if(name != '') {
		var elems = document.getElementsByName(name);
		for (var n = 0; n < elems.length; n++)
			elems[n].disabled = true;
	}
}

function placeFocus() {
	/* place focus on first element of first form. */
	document.forms[0].elements[0].focus();
}

function triggerForm(name, val) {
	/* call the onChange event of the form element with 'name' and value 'val',
	 * so that the appropriate form elements may be shown/hidden. */
	var elems = document.getElementsByName(name)
	for (var n = 0; n < elems.length; n++) {
		if(elems[n].value == val) {
			elems[n].onchange();
			break;
		}
	}
}

function highlightdep(id) {
	['word', 'tag', 'dependency', 'edge', 'arrow'].forEach(function(a) {
		var elems = document.getElementsByClassName(a);
		for (var n = 0; n < elems.length; n++)
			elems[n].style = '';
	});
	var elems = document.getElementsByClassName(id);
	for (var n = 0; n < elems.length; n++)
		elems[n].style = 'stroke: black !important; ';
}

function nohighlightdep() {
	['word', 'tag', 'dependency', 'edge', 'arrow'].forEach(function(a) {
		var elems = document.getElementsByClassName(a);
		for (var n = 0; n < elems.length; n++)
			elems[n].style = '';
	});
}

function annotate() {
	/* function to send request to parse a sentence and append the result to
	 * the current document. */
	var xmlhttp = getxmlhttp();
	var div = document.getElementById('result');
	div.innerHTML = '[...wait for it...]';
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			div.innerHTML = xmlhttp.responseText;
			registertoggleable(div);
		}
	};
	url = "/annotate/parse?html=1&sent=" + encodeURIComponent(document.queryform.sent.value);
	/* if there were any filter constraints, convert them to parsing constraints now */
	require.push.apply(require, frequire);
	block.push.apply(block, fblock);
	frequire = [];
	fblock = [];
	if(require.length > 0 || block.length > 0) {
		url += "&require=" + encodeURIComponent(require.join('\t'))
				+ "&block=" + encodeURIComponent(block.join('\t'));
		document.getElementById('constraintdiv').style.display = 'block';
	}
	url += '&sentno=' + document.queryform.sentno.value;
	xmlhttp.open("GET", url, true);
	xmlhttp.send(null);
}


/* constraints used during parsing */
var require = [];
var block = [];
/* constraints used only for filtering */
var frequire = [];
var fblock = [];
function togglespan(flag, pos, elem) {
	document.getElementById('constraintdiv').style.display = 'block';
	var item = elem.dataset.s;
	/* flag=0: make span required; flag=1: block span. */
	if(flag == 0) {
		array1 = frequire;
		array2 = fblock;
	} else {
		array1 = fblock;
		array2 = frequire;
	}
	var i = array2.indexOf(item);
	if(i != -1) {
		array2.splice(i, 1);  // remove array2[i]
		var elems = document.querySelectorAll('#showrequire > span, #showblock > span');
		for (var n = 0; n < elems.length; n++) {
			if(typeof elems[n].dataset !== 'undefined' && elems[n].dataset.s == item) {
				elems[n].parentNode.removeChild(elems[n]);
			}
		}
	}
	i = array1.indexOf(item);
	if(i != -1) {
		array1.splice(i, 1);  // remove array1[i]
		elem.style.backgroundColor = 'white';
		var elems = document.querySelectorAll('#showrequire > span, #showblock > span');
		for (var n = 0; n < elems.length; n++) {
			if(typeof elems[n].dataset !== 'undefined' && elems[n].dataset.s == item) {
				elems[n].parentNode.removeChild(elems[n]);
			}
		}
	} else {
		var elem1 = document.createElement('span');
		elem1.innerHTML = item;
		elem1.dataset.s = elem.dataset.s;
		if(flag == 0) {
			elem1.style.backgroundColor = 'lightgreen';
			elem1.onclick = function(){ togglespan(0, pos, elem1); };
			document.getElementById('showrequire').appendChild(elem1);
			document.getElementById('showrequire').appendChild(document.createTextNode(' '));
		} else {
			elem1.style.backgroundColor = 'lightcoral';
			elem1.onclick = function(){ togglespan(1, pos, elem1); };
			document.getElementById('showblock').appendChild(elem1);
			document.getElementById('showblock').appendChild(document.createTextNode(' '));
		}
		array1.push(item); 	// append item to array1
	}

	// make AJAX call to display only matching trees
	var xmlhttp = getxmlhttp();
	var div = document.getElementById('nbest');
	if(div.style.display == 'none')
		div.style.display = 'block';
	div.innerHTML = '[...wait for it...]';

	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			div.innerHTML = xmlhttp.responseText;
			registertoggleable(div);
			var elems = div.querySelectorAll('span.n, span.p');
			for (var n = 0; n < elems.length; n++) {
				if(typeof elems[n].dataset !== 'undefined') {
					if(require.indexOf(elems[n].dataset.s) != -1
							|| frequire.indexOf(elems[n].dataset.s) != -1) {
						elems[n].style.backgroundColor = 'lightgreen';
					} else if(block.indexOf(elems[n].dataset.s) != -1
							|| fblock.indexOf(elems[n].dataset.s) != -1) {
						elems[n].style.backgroundColor = 'lightcoral';
					}
				}
			}
		}
	};
	var lang = document.queryform.lang;
	url = "/annotate/filter?sent=" + encodeURIComponent(document.queryform.sent.value);
	if(require.length > 0 || block.length > 0)
		url += "&require=" + encodeURIComponent(require.join('\t'))
			+ "&block=" + encodeURIComponent(block.join('\t'));
	url += "&frequire=" + encodeURIComponent(frequire.join('\t'))
		+ "&fblock=" + encodeURIComponent(fblock.join('\t'))
		+ '&sentno=' + document.queryform.sentno.value;
	xmlhttp.open("GET", url, true);
	xmlhttp.send(null);

	return false;  // do not handle click further
}

function registertoggleable(div) {
	var elems = div.getElementsByClassName('n');
	for (var n = 0; n < elems.length; n++) {
		elems[n].onclick = function(event) { togglespan(0, 0, event.currentTarget); }
		elems[n].oncontextmenu = function(event) { togglespan(1, 0, event.currentTarget); event.preventDefault(); }
	}
	var elems = div.getElementsByClassName('p');
	for (var n = 0; n < elems.length; n++) {
		elems[n].onclick = function(event) { togglespan(0, 1, event.currentTarget); }
		elems[n].oncontextmenu = function(event) { togglespan(1, 1, event.currentTarget); event.preventDefault(); }
	}
}

function registerdraggable(div) {
	var elems = div.getElementsByClassName('n');
	for (var n = 0; n < elems.length; n++) {
		elems[n].onclick = pickphrasal;
		elems[n].oncontextmenu = reparsesubtree;
		elems[n].draggable = true;
		elems[n].ondragstart = drag;
		elems[n].style = "cursor: move;"
		elems[n].ondrop = drop;
		elems[n].ondragover = allowDrop;
	}
	var elems = div.getElementsByClassName('p');
	console.log(elems)
	for (var n = 0; n < elems.length; n++) {
		if (elems[n].attributes.editable.value == 'true') {
			elems[n].onclick = pickpos;
			elems[n].draggable = true;
			elems[n].ondragstart = drag;
			elems[n].style = "cursor: move;"
		}
	}
	var elems = div.getElementsByClassName('f');
	for (var n = 0; n < elems.length; n++) {
		if (elems[n].attributes.editable.value == 'true') {
			elems[n].onclick = pickfunction;
		}
		// elems[n].draggable = true;
		// elems[n].ondragstart = drag;
		// elems[n].style = "cursor: move;"
	}
	var elems = div.getElementsByClassName('m');
	for (var n = 0; n < elems.length; n++) {
		elems[n].onclick = pickmorph;
	}
}

function replacetree() {
	// make AJAX call to visualize edited tree.
	var xmlhttp = getxmlhttp();
	var el = document.getElementById('tree');
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			resp = JSON.parse(xmlhttp.responseText)
			el.innerHTML = resp.html;
			registerdraggable(el);
			if (! resp.has_error) {
				// Update oldtree with the current value from the editor
				oldtree = editor.getValue();
			}
		}
	};
	// Create the data object to be sent in a POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		oldtree: oldtree,
		tree: editor.getValue()
	};

	// Convert the data object to a JSON string
	const jsonData = JSON.stringify(data);

	// Open the POST request
	xmlhttp.open("POST", '/annotate/redraw', true);

	// Set the request header to indicate the content type
	xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

	// Send the JSON data in the body of the request
	xmlhttp.send(jsonData);
}

// global to track which node is being modified when label picker is used
var nodeid = '';

function pickphrasal(ev) {
	var modifier = ev.ctrlKey || ev.metaKey
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'phrasalpicker');
	}
}
function pickpos(ev) {
	var modifier = ev.ctrlKey || ev.metaKey
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'pospicker');
	}
}
function pickfunction(ev) {
	ev.stopPropagation();  // block pickphrasal() from triggering
	var modifier = ev.ctrlKey || ev.metaKey
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'functionpicker');
	}
}
function pickmorph(ev) {
	ev.stopPropagation();  // block pickphrasal() from triggering
	var modifier = ev.ctrlKey || ev.metaKey
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'morphpicker');
	}
}
function showpicker(ev, picker) {
	// show pop-up menu to select a different label for given node
	var node = ev.currentTarget;
	// change global so that label is submitted for this node
	nodeid = node.dataset.id;
	picker = document.getElementById(picker);
	// move picker above node:
	var rect = node.getClientRects()[0];
	picker.style.top = rect.top + 'px';
	if (rect.left > window.innerWidth / 2) {
		var width = 0.3 * window.innerWidth;
		picker.style.left = (rect.right - width) + 'px';
	} else {
		picker.style.left = rect.left + 'px';
	}
	picker.style.display = 'block';  // show picker
	return false;
}

function pick(labeltype, label) {
	// make AJAX call to visualize tree with a newly picked label.
	// hide div
	document.getElementById(labeltype + 'picker').style.display = 'none';
	if (label === null) {
		return;
	}
	var xmlhttp = getxmlhttp();
	var el = document.getElementById('tree');
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			var res = xmlhttp.responseText.split('\t', 2);
			el.innerHTML = res[0];
			if(res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el);
		}
	};
	// Create the data object to be sent in the POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		nodeid: nodeid,
		tree: editor.getValue()
	};

	// Add the appropriate label based on the labeltype
	if (labeltype == 'function') {
		data.function = label;
	} else if (labeltype == 'morph') {
		data.morph = label;
	} else {
		data.label = label;
	}

	// Convert the data object to a JSON string
	const jsonData = JSON.stringify(data);

	// Open the POST request
	xmlhttp.open("POST", '/annotate/newlabel', true);

	// Set the request header to indicate the content type
	xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

	// Send the JSON data in the body of the request
	xmlhttp.send(jsonData);
}

function reparsesubtree(ev) {
	ev.stopPropagation();
	var node = ev.currentTarget;
	var xmlhttp = getxmlhttp();
	// change global so that label is submitted for this node
	nodeid = node.dataset.id;
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			var res = xmlhttp.responseText.split('\t', 2);
			var el = document.getElementById('nbest');
			el.innerHTML = res[0];
			el.style.display = 'block';
			if(res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
		}
	};
	url = '/annotate/reparsesubtree?sentno=' + document.queryform.sentno.value
			+ '&nodeid=' + encodeURIComponent(node.dataset.id)
			+ '&tree=' + encodeURIComponent(editor.getValue());
	xmlhttp.open("GET", url, true);
	xmlhttp.send(null);
	return false;
}

function picksubtree(n) {
	var xmlhttp = getxmlhttp();
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			var res = xmlhttp.responseText.split('\t', 2);
			var el = document.getElementById('tree');
			el.innerHTML = res[0];
			if(res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el);
			var el = document.getElementById('nbest');
			el.innerHTML = '';
			el.style.display = 'none';
		}
	};
	url = '/annotate/replacesubtree?sentno=' + document.queryform.sentno.value
			+ '&n=' + n
			+ '&nodeid=' + encodeURIComponent(nodeid)
			+ '&tree=' + encodeURIComponent(editor.getValue());
	xmlhttp.open("GET", url, true);
	xmlhttp.send(null);
	return false;
}

function drag(ev) {
    ev.dataTransfer.setData("text", ev.target.dataset.id);
}

function allowDrop(ev) {
	ev.preventDefault();
}

function allowDropTrash(ev) {
	allowDrop(ev);
	ev.target.style.opacity=1;
}

function dragLeaveTrash(ev) {
	ev.preventDefault();
	ev.target.style.opacity=.5;
}

function dropTrash(ev) {
	drop(ev);
	ev.target.style.opacity=.5;
}

function drop(ev) {
	/* request tree where dragged "childid" is re-attached under "newparentid". */
    ev.preventDefault();
    var childid = ev.dataTransfer.getData("text");
	var newparentid = ev.target.dataset.id;
	var xmlhttp;
	if(window.XMLHttpRequest) {
		// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();
	} else if(window.ActiveXObject) {
		// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	} else {
		alert("Your browser does not support XMLHTTP!");
	}
	var el = document.getElementById('tree');
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			var res = xmlhttp.responseText.split('\t', 2);
			el.innerHTML = res[0];
			if(res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el);
		}
	};
	// Create the data object to be sent in the POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		nodeid: childid,
		newparent: newparentid,
		tree: editor.getValue()
	};

	// Convert the data object to a JSON string
	const jsonData = JSON.stringify(data);

	// Open the POST request
	xmlhttp.open("POST", '/annotate/reattach', true);

	// Set the request header to indicate the content type
	xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

	// Send the JSON data in the body of the request
	xmlhttp.send(jsonData);
}

function newproj(ev) {
	/* request tree where dragged "childid" is re-attached under "newparentid". */
    ev.preventDefault();
    // var childid = ev.dataTransfer.getData("text");
	var targetid = ev.target.dataset.id;
	var xmlhttp;
	if(window.XMLHttpRequest) {
		// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();
	} else if(window.ActiveXObject) {
		// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	} else {
		alert("Your browser does not support XMLHTTP!");
	}
	var el = document.getElementById('tree');
	xmlhttp.onreadystatechange=function() {
		if(xmlhttp.readyState==4) { // && xmlhttp.status==200) {
			var res = xmlhttp.responseText.split('\t', 2);
			el.innerHTML = res[0];
			if(res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el);
		}
	};
	// Create the data object to be sent in the POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		nodeid: 'newproj',
		newparent: targetid,
		tree: editor.getValue()
	};

	// Convert the data object to a JSON string
	const jsonData = JSON.stringify(data);

	// Open the POST request
	xmlhttp.open("POST", '/annotate/reattach', true);

	// Set the request header to indicate the content type
	xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

	// Send the JSON data in the body of the request
	xmlhttp.send(jsonData);
}

function accept() {
	// make AJAX call to accept the current tree.
	var xmlhttp = getxmlhttp();
	var tree = editor.getValue();
	var sentno = document.getElementById('sentno').value;
	
	// Create the data object to be sent in the POST request
	const data = {
		sentno: sentno,
		tree: tree,
	};

	console.log(data);

	// Convert the data object to a JSON string
	const jsonData = JSON.stringify(data);

	// Define the callback function to handle the response
	  xmlhttp.onreadystatechange = function() {
        if (xmlhttp.readyState === 4) {
            if (xmlhttp.status === 200 || xmlhttp.status === 302) {
                var responseURL = xmlhttp.responseURL;
                if (responseURL) {
                    // Redirect the user to the specified URL
                    window.location.href = responseURL;
                } else {
                    console.error('No redirect URL found in the response');
                }
            } else {
                console.error('Error: ' + xmlhttp.status);
            }
        }
    };

	// Open the POST request
	xmlhttp.open("POST", '/annotate/accept', true);

	// Set the request header to indicate the content type
	xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

	// Send the JSON data in the body of the request
	xmlhttp.send(jsonData);
}

function addSentence() {
	// Make a GET request to the server to generate a unique hash id for the new sentence (and populate the #idInput field with it)
	$.ajax({
		url: "/annotate/get_id",
		type: "GET",
		success: function(response, textStatus, jqXHR) {
			var id = response.id;
			$('#idInput').val(id);
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
	// Display the dialog box for adding a new sentence
	$("#sentEntry").dialog({
		modal: true,
		buttons: {
			"Submit": function() {
				var sent = $('#sentenceInput').val();
				var id = $('#idInput').val();
				if (sent !== null) {
					// Process the user input
					console.log("User entered: " + sent);
					
					// Make the AJAX GET request using jQuery
					$.ajax({
						url: "/annotate/direct_entry",
						type: "GET",
						data: { sent: sent, id: id },
						success: function(response, textStatus, jqXHR) {
							var responseURL = response.redirect_url;
							var responseError = response.error;
							// Handle the redirect
							if (responseURL) {
								// Redirect the user to the specified URL
								window.location.href = responseURL;
								$(this).dialog("close");
							} else if ( responseError ) {
								$("#entryError").show();
								$("#entryError").text("Error: " + responseError);
							}
						},
						error: function(jqXHR, textStatus, errorThrown) {
							console.error('Error: ' + jqXHR.status);
						}
					});
				}
			},
			"Cancel": function() {
				$(this).dialog("close");
			}
		}
	});
};