SELECT DISTINCT ar.object_name
FROM act_relationships ar
LEFT JOIN legislations l ON ar.object_name = l.title
WHERE l.title IS NULL;
