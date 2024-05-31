-- Create the database
CREATE DATABASE Pizza_db;
USE Pizza_db;
 
-- Create the Pizza table
CREATE TABLE Pizza (
    pizza_id INT PRIMARY KEY,
    pizza_name VARCHAR(30) NOT NULL,
    pizza_category VARCHAR(30) NOT NULL,
    size VARCHAR(10) NOT NULL,
    price INT NOT NULL,
    stock_sold INT, 
    UNIQUE KEY name_category_size (pizza_name, pizza_category, size)
);
 
-- Create the discounts table
CREATE TABLE discounts (
    discount_id INT AUTO_INCREMENT PRIMARY KEY,
    pizza_id INT NOT NULL,
    pct_discount INT ,
    FOREIGN KEY (pizza_id) REFERENCES Pizza(pizza_id)
);
 
-- Insert sample records to Pizza table
insert into pizza values(	1	,	'Deluxe Veggie',	'Veg',	'XS',	50	,	5	);
insert into pizza values(	2	,	'Deluxe Veggie',	'Veg',	'S',	100	,	10	);
insert into pizza values(	3	,	'Deluxe Veggie',	'Veg',	'M',	150	,	7	);
insert into pizza values(	4	,	'Deluxe Veggie',	'Veg',	'L',	200	,	13	);
insert into pizza values(	5	,	'Deluxe Veggie',	'Veg',	'XL',	250	,	18	);
insert into pizza values(	6	,	'Farmhouse',	'Veg',	'XS',	100	,	8	);
insert into pizza values(	7	,	'Farmhouse',	'Veg',	'S',	200	,	12	);
insert into pizza values(	8	,	'Farmhouse',	'Veg',	'M',	300	,	6	);
insert into pizza values(	9	,	'Farmhouse',	'Veg',	'L',	400	,	14	);
insert into pizza values(	10	,	'Farmhouse',	'Veg',	'XL',	500	,	21	);
insert into pizza values(	11	,	'Barbeque Chicken',	'Non-Veg',	'XS',	250	,	3	);
insert into pizza values(	12	,	'Barbeque Chicken',	'Non-Veg',	'S',	300	,	11	);
insert into pizza values(	13	,	'Barbeque Chicken',	'Non-Veg',	'M',	450	,	8	);
insert into pizza values(	14	,	'Barbeque Chicken',	'Non-Veg',	'L',	550	,	15	);
insert into pizza values(	15	,	'Barbeque Chicken',	'Non-Veg',	'XL',	650	,	22	);
insert into pizza values(	16	,	'Chicken Golden Delight',	'Non-Veg',	'XS',	120	,	9	);
insert into pizza values(	17	,	'Chicken Golden Delight',	'Non-Veg',	'S',	200	,	7	);
insert into pizza values(	18	,	'Chicken Golden Delight',	'Non-Veg',	'M',	250	,	8	);
insert into pizza values(	19	,	'Chicken Golden Delight',	'Non-Veg',	'L',	400	,	14	);
insert into pizza values(	20	,	'Chicken Golden Delight',	'Non-Veg',	'XL',	600	,	28	);

-- Insert sample records to Discounts table
insert into discounts values(	1	,	4	,	10	);
insert into discounts values(	2	,	5	,	15	);
insert into discounts values(	3	,	9	,	10	);
insert into discounts values(	4	,	10	,	15	);
insert into discounts values(	5	,	14	,	20	);
insert into discounts values(	6	,	15	,	30	);
insert into discounts values(	7	,	19	,	25	);
insert into discounts values(	8	,	20	,	30	);