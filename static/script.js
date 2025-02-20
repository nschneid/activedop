function toggle(id) {
	/* toggle element with id to be hidden or not. */
	var el = $('#' + id);
	el.toggle();
}

function showhide(id1, id2, id3, n) {
	/* show id1, hide id2; hide dd0 .. dd<n>, but show id3*/
	$('#' + id1).show();
	$('#' + id2).hide();
	for (var i = 0; i < n; i++) {
		$('#dd' + i).hide();
	}
	$('#' + id3).show();
}

function togglelink(id) {
	/* toggle element with id to be hidden or not, and also toggle
	 * link with id 'link'+id to start with 'show' or 'hide'. */
	var el = $('#' + id);
	var link = $('#link' + id);
	el.toggle();
	if(el.is(':visible')) {
		link.html('hide' + link.html().substring(4));
	} else {
		link.html('show' + link.html().substring(4));
	}
}

function toggletextbox() {
	/* toggle a textbox to be single or multi line. */
	var state = $('#textarea');
	var cur = $('#query');
	var next = $('#notquery');
	var link = $('#textboxlink');
	cur.prop('name', 'notquery').prop('disabled', true).hide();
	next.prop('name', 'query').prop('disabled', false).show();
	if(state.prop('disabled')) {
		state.prop('disabled', false);
		next.val(cur.val());
		link.html('smaller');
	} else {
		state.prop('disabled', true);
		cur.val(next.val());
		link.html('larger');
	}
}

function show(id, name) {
	/* show element with 'id' and enable (un-disable) all form elements with 'name'. */
	var el = $('#' + id);
	el.css('visibility', 'visible');
	if(name != '') {
		var elems = $('[name="' + name + '"]');
		elems.prop('disabled', false);
	}
}

function hide(id, name) {
	/* hide element with 'id' and disable all form elements with 'name'. */
	var el = $('#' + id);
	el.css('visibility', 'hidden');
	if(name != '') {
		var elems = $('[name="' + name + '"]');
		elems.prop('disabled', true);
	}
}

function placeFocus() {
	/* place focus on first element of first form. */
	$('form:first :input:first').focus();
}

function triggerForm(name, val) {
	/* call the onChange event of the form element with 'name' and value 'val',
	 * so that the appropriate form elements may be shown/hidden. */
	var elems = $('[name="' + name + '"]');
	for (var n = 0; n < elems.length; n++) {
		if(elems[n].value == val) {
			$(elems[n]).change();
			break;
		}
	}
}

function highlightdep(id) {
	['word', 'tag', 'dependency', 'edge', 'arrow'].forEach(function(a) {
		var elems = $('.' + a);
		elems.css('stroke', '');
	});
	var elems = $('.' + id);
	elems.css('stroke', 'black !important');
}

function nohighlightdep() {
	['word', 'tag', 'dependency', 'edge', 'arrow'].forEach(function(a) {
		var elems = $('.' + a);
		elems.css('stroke', '');
	});
}

function annotate() {
	/* function to send request to parse a sentence and append the result to
	 * the current document. */
	var div = $('#result');
	div.html('[...wait for it...]');

	// Construct the URL for the AJAX request
	var url = "/annotate/parse?html=1&sent=" + encodeURIComponent(document.queryform.sent.value);

	/* if there were any filter constraints, convert them to parsing constraints now */
	require.push.apply(require, frequire);
	block.push.apply(block, fblock);
	frequire = [];
	fblock = [];
	if(require.length > 0 || block.length > 0) {
		url += "&require=" + encodeURIComponent(require.join('\t'))
				+ "&block=" + encodeURIComponent(block.join('\t'));
		$('#constraintdiv').show();
	}
	url += '&sentno=' + document.queryform.sentno.value;

	// Make the AJAX GET request using jQuery
	$.ajax({
		url: url,
		type: "GET",
		success: function(response) {
			div.html(response);
			registertoggleable(div[0]);
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
}

/* constraints used during parsing */
var require = [];
var block = [];
/* constraints used only for filtering */
var frequire = [];
var fblock = [];
function togglespan(flag, pos, elem) {
	$('#constraintdiv').show();
	var item = $(elem).data('s');
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
		$('#showrequire > span, #showblock > span').each(function() {
			if($(this).data('s') == item) {
				$(this).remove();
			}
		});
	}
	i = array1.indexOf(item);
	if(i != -1) {
		array1.splice(i, 1);  // remove array1[i]
		$(elem).css('background-color', 'white');
		$('#showrequire > span, #showblock > span').each(function() {
			if($(this).data('s') == item) {
				$(this).remove();
			}
		});
	} else {
		var elem1 = $('<span>').html(item).data('s', $(elem).data('s'));
		if(flag == 0) {
			elem1.css('background-color', 'lightgreen').click(function() { togglespan(0, pos, elem1); });
			$('#showrequire').append(elem1).append(' ');
		} else {
			elem1.css('background-color', 'lightcoral').click(function() { togglespan(1, pos, elem1); });
			$('#showblock').append(elem1).append(' ');
		}
		array1.push(item); 	// append item to array1
	}

	// make AJAX call to display only matching trees
	var div = $('#nbest');
	if(div.css('display') == 'none')
		div.show();
	div.html('[...wait for it...]');

	$.ajax({
		url: "/annotate/filter",
		type: "GET",
		data: {
			sent: document.queryform.sent.value,
			require: require.join('\t'),
			block: block.join('\t'),
			frequire: frequire.join('\t'),
			fblock: fblock.join('\t'),
			sentno: document.queryform.sentno.value
		},
		success: function(response) {
			div.html(response);
			registertoggleable(div[0]);
			var elems = div.find('span.n, span.p');
			elems.each(function() {
				if(require.indexOf($(this).data('s')) != -1
						|| frequire.indexOf($(this).data('s')) != -1) {
					$(this).css('background-color', 'lightgreen');
				} else if(block.indexOf($(this).data('s')) != -1
						|| fblock.indexOf($(this).data('s')) != -1) {
					$(this).css('background-color', 'lightcoral');
				}
			});
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});

	return false;  // do not handle click further
}

function registertoggleable(div) {
	var elems = $(div).find('.n');
	elems.each(function() {
		$(this).on('click', function(event) { togglespan(0, 0, event.currentTarget); });
		$(this).on('contextmenu', function(event) { togglespan(1, 0, event.currentTarget); event.preventDefault(); });
	});
	elems = $(div).find('.p');
	elems.each(function() {
		$(this).on('click', function(event) { togglespan(0, 1, event.currentTarget); });
		$(this).on('contextmenu', function(event) { togglespan(1, 1, event.currentTarget); event.preventDefault(); });
	});
}

function registerdraggable(div) {
	var elems = $(div).find('.n');
	elems.each(function() {
		$(this).on('click', pickphrasal);
		$(this).on('contextmenu', reparsesubtree);
		$(this).attr('draggable', true);
		$(this).on('dragstart', drag);
		$(this).css('cursor', 'move');
		$(this).on('drop', drop);
		$(this).on('dragover', allowDrop);
	});
	elems = $(div).find('.p');
	elems.each(function() {
		if ($(this).attr('editable') == 'true') {
			$(this).on('click', pickpos);
			$(this).attr('draggable', true);
			$(this).on('dragstart', drag);
			$(this).css('cursor', 'move');
		}
	});
	elems = $(div).find('.f');
	elems.each(function() {
		if ($(this).attr('editable') == 'true') {
			$(this).on('click', pickfunction);
		}
	});
	elems = $(div).find('.m');
	elems.each(function() {
		$(this).on('click', pickmorph);
	});
}

function replacetree() {
	var el = $('#tree');
	$.ajax({
		url: '/annotate/redraw',
		type: 'POST',
		contentType: 'application/json',
		data: JSON.stringify({
			sentno: document.queryform.sentno.value,
			senttok: document.queryform.senttok.value,
			oldtree: oldtree,
			tree: editor.getValue()
		}),
		success: function(response) {
			el.html(response.html);
			registerdraggable(el[0]);
			if (!response.has_error) {
				// Update oldtree with the current value from the editor
				oldtree = editor.getValue();
			}
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
}

function pickphrasal(ev) {
	var modifier = ev.ctrlKey || ev.metaKey;
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'phrasalpicker');
	}
}

function pickpos(ev) {
	var modifier = ev.ctrlKey || ev.metaKey;
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'pospicker');
	}
}

function pickfunction(ev) {
	ev.stopPropagation();
	var modifier = ev.ctrlKey || ev.metaKey;
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'functionpicker');
	}
}

function pickmorph(ev) {
	ev.stopPropagation();
	var modifier = ev.ctrlKey || ev.metaKey;
	if (modifier) {
		return newproj(ev);
	} else {
		return showpicker(ev, 'morphpicker');
	}
}

function showpicker(ev, picker) {
	var node = ev.currentTarget;
	nodeid = $(node).data('id');
	picker = $('#' + picker);
	var rect = node.getBoundingClientRect();
	picker.css('top', rect.top + 'px');
	if (rect.left > window.innerWidth / 2) {
		var width = 0.3 * window.innerWidth;
		picker.css('left', (rect.right - width) + 'px');
	} else {
		picker.css('left', rect.left + 'px');
	}
	picker.show();
	return false;
}

function pick(labeltype, label) {
	$('#' + labeltype + 'picker').hide();
	if (label === null) {
		return;
	}
	var el = $('#tree');

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

	// Make the AJAX POST request using jQuery
	$.ajax({
		url: '/annotate/newlabel',
		type: 'POST',
		contentType: 'application/json',
		data: JSON.stringify(data),
		success: function(response) {
			var res = response.split('\t', 2);
			el.html(res[0]);
			if (res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el[0]);
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
}

function reparsesubtree(ev) {
	ev.stopPropagation();
	var node = ev.currentTarget;
	nodeid = $(node).data('id');
	$.ajax({
		url: '/annotate/reparsesubtree',
		type: 'GET',
		data: {
			sentno: document.queryform.sentno.value,
			nodeid: nodeid,
			tree: editor.getValue()
		},
		success: function(response) {
			var res = response.split('\t', 2);
			var el = $('#nbest');
			el.html(res[0]);
			el.show();
			if (res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
	return false;
}

function picksubtree(n) {
	$.ajax({
		url: '/annotate/replacesubtree',
		type: 'GET',
		data: {
			sentno: document.queryform.sentno.value,
			n: n,
			nodeid: nodeid,
			tree: editor.getValue()
		},
		success: function(response) {
			var res = response.split('\t', 2);
			var el = $('#tree');
			el.html(res[0]);
			if (res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el[0]);
			el = $('#nbest');
			el.html('');
			el.hide();
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
	return false;
}

function drag(ev) {
	ev.originalEvent.dataTransfer.setData("text", $(ev.target).data('id'));
}

function allowDrop(ev) {
	ev.preventDefault();
}

function allowDropTrash(ev) {
	allowDrop(ev);
	$(ev.target).css('opacity', 1);
}

function dragLeaveTrash(ev) {
	ev.preventDefault();
	$(ev.target).css('opacity', 0.5);
}

function dropTrash(ev) {
	drop(ev);
	$(ev.target).css('opacity', 0.5);
}

function drop(ev) {
	ev.preventDefault();
	var childid = (ev.originalEvent || ev).dataTransfer.getData("text");
	var newparentid = $(ev.target).data('id');
	var el = $('#tree');

	// Create the data object to be sent in the POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		nodeid: childid,
		newparent: newparentid,
		tree: editor.getValue()
	};

	// Make the AJAX POST request using jQuery
	$.ajax({
		url: '/annotate/reattach',
		type: 'POST',
		contentType: 'application/json',
		data: JSON.stringify(data),
		success: function(response) {
			var res = response.split('\t', 2);
			el.html(res[0]);
			if (res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el[0]);
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
}

function newproj(ev) {
	ev.preventDefault();
	var targetid = $(ev.target).data('id');
	var el = $('#tree');

	// Create the data object to be sent in the POST request
	const data = {
		sentno: document.queryform.sentno.value,
		senttok: document.queryform.senttok.value,
		nodeid: 'newproj',
		newparent: targetid,
		tree: editor.getValue()
	};

	// Make the AJAX POST request using jQuery
	$.ajax({
		url: '/annotate/reattach',
		type: 'POST',
		contentType: 'application/json',
		data: JSON.stringify(data),
		success: function(response) {
			var res = response.split('\t', 2);
			el.html(res[0]);
			if (res[1]) {
				editor.setValue(res[1]);
				oldtree = editor.getValue();
			}
			registerdraggable(el[0]);
		},
		error: function(jqXHR, textStatus, errorThrown) {
			console.error('Error: ' + jqXHR.status);
		}
	});
}

function accept() {
	// Create the data object to be sent in the POST request
	const data = {
		sentno: $('#sentno').val(),
		tree: editor.getValue()
	};

	// Make the AJAX POST request using jQuery
	$.ajax({
		url: '/annotate/accept',
		type: 'POST',
		contentType: 'application/json',
		data: JSON.stringify(data),
		success: function(response, textStatus, jqXHR) {
			var responseURL = response.redirect_url;
			var responseError = response.error;
			// Handle the redirect
			if (responseURL) {
				// Redirect the user to the specified URL
				window.location.href = responseURL;
			} else if ( responseError ) {
				console.error("Error: " + responseError);
			}
		},
		error: function(jqXHR) {
			console.error('Error: ' + jqXHR.status);
		}
	});
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