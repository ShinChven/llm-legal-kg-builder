select
	sum(case when source = 'legislation.govt.nz' then 1 else 0 end) as acts_from_legislation_govt_nz_subscribe,
	sum(case when source = 'www.legislation.govt.nz' then 1 else 0 end) as acts_from_legislation_govt_nz_html,
	sum(case when source = 'nzlii.org' then 1 else 0 end) as acts_from_nzlii_org
from legislations
where (source in ('legislation.govt.nz', 'www.legislation.govt.nz', 'nzlii.org') or text is null)
	 or word_count > 0;
