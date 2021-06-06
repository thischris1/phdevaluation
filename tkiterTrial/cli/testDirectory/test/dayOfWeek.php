<?php
$wochentage = array("Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag");
$zeit = strtotime(date('y-m-d'));
echo $zeit;
echo $wochentage[date("w", $zeit)];
if ( $wochentage[date("w", $zeit)] == 'Sonntag') {
	echo "Wochenende";
}
if ( $wochentage[date("w", $zeit)] == 'Samstag') {
	echo "Wochenende";
}
if ( $wochentage[date("w", $zeit)] == 'Montag') {
	echo "Montag - Wochenende berückischtigen";
}

?>
