# Haveli Harvest Oils Pvt. Ltd.: company brief

> **Fictional sample company** for the Forecast Engine v2 mock run. The company, its brands,
> people and customers are invented. The sales history is synthetic, but it reacts to real
> events: real festival dates, real New Delhi temperatures and India's real edible-oil import
> duty changes (see [What is real](#what-is-real-and-what-is-invented)).

## At a glance

| | |
| --- | --- |
| Business | Edible-oil crushing, refining and packing |
| Founded | 1998, family-owned |
| Plants | Bharatpur, Rajasthan: mustard expeller, refinery, packing. Ghaziabad, UP (Delhi NCR): second packing unit |
| Markets | Rajasthan, Delhi NCR, Haryana, Uttar Pradesh |
| Size | About 5.25 million litres/kg a year, about ₹74 crore at distributor prices |
| Brands | HH Kachi Ghani (mustard), HH Gold (soybean), HH Rasoi (palmolein), HH Sunlite (sunflower), HH Heart (rice bran), HH Vanaspati |
| ERP data | Monthly sales by customer and item, Nov 2023 to Oct 2026: 15 items, 5 customers, 59 customer-item series |
| Client id in the engine | `demo_haveliharvest` |

## Products

15 items in six item groups. Prices are per unit (one pouch, bottle, jar or tin) and
illustrative.

| Item code | Product | Group | Sales unit | MRP ₹ | Main raw material and origin |
| --- | --- | --- | --- | --- | --- |
| HH-MUS-1LP | Kachi Ghani Mustard Oil 1 L pouch | MUST | Carton of 12 | 185 | Mustard seed: Rajasthan, Haryana, MP |
| HH-MUS-1LB | Kachi Ghani Mustard Oil 1 L bottle | MUST | Carton of 12 | 199 | Mustard seed |
| HH-MUS-5LJ | Kachi Ghani Mustard Oil 5 L jar | MUST | Carton of 4 | 920 | Mustard seed |
| HH-MUS-15T | Kachi Ghani Mustard Oil 15 kg tin | MUST | Tin | 2,650 | Mustard seed |
| HH-SOY-1LP | Gold Refined Soybean Oil 1 L pouch | SOYA | Carton of 12 | 150 | Crude soybean oil: Argentina, Brazil (~70%), MP (~30%) |
| HH-SOY-5LJ | Gold Refined Soybean Oil 5 L jar | SOYA | Carton of 4 | 740 | Crude soybean oil |
| HH-SOY-15T | Gold Refined Soybean Oil 15 kg tin | SOYA | Tin | 2,250 | Crude soybean oil |
| HH-PAL-1LP | Rasoi Refined Palmolein 1 L pouch | PALM | Carton of 12 | 132 | Crude palm oil: Indonesia, Malaysia |
| HH-PAL-15T | Rasoi Refined Palmolein 15 kg tin | PALM | Tin | 1,990 | Crude palm oil |
| HH-SUN-1LP | Sunlite Refined Sunflower Oil 1 L pouch | SUNF | Carton of 12 | 165 | Crude sunflower oil: Russia, Ukraine (Black Sea) |
| HH-SUN-5LJ | Sunlite Refined Sunflower Oil 5 L jar | SUNF | Carton of 4 | 810 | Crude sunflower oil |
| HH-RBO-1LB | Heart Rice Bran Oil 1 L bottle | RICE | Carton of 12 | 175 | Crude rice bran oil: Punjab, Haryana rice mills |
| HH-RBO-5LJ | Heart Rice Bran Oil 5 L jar | RICE | Carton of 4 | 860 | Crude rice bran oil |
| HH-VAN-1KP | Vanaspati 1 kg pouch | VANA | Carton of 12 | 155 | Palm oil and palm stearin (imported) |
| HH-VAN-15T | Vanaspati 15 kg tin | VANA | Tin | 2,240 | Palm oil and palm stearin |

The full item master, with distributor prices, packaging and buying pattern, is in
[`data/demo/haveliharvest/item_master.csv`](../../data/demo/haveliharvest/item_master.csv).

**Volume mix (last 12 months):** mustard 37%, soybean 26%, palmolein 19%, vanaspati 8%,
sunflower 6%, rice bran 4%. **About 59% of volume is made from imported crude oil.**

## Customers

| Customer | Channel | Base | Share of volume |
| --- | --- | --- | --- |
| Yamuna Food Distributors | Distributor | New Delhi | 31% |
| Shekhawati Traders | Distributor | Jaipur | 25% |
| Awadh Provisions | Distributor | Lucknow | 24% |
| Halwai and Caterers Desk | Institutional: sweet shops, caterers, namkeen makers (15 kg tins only) | Jaipur, Agra, Delhi | 11% |
| BazaarKart Supermarkets | Modern trade (retail packs only, growing about 16% a year) | Gurugram | 10% |

## How the business works

- **Buying raw material.** Imported crude oils arrive at Kandla port and are booked 6–8
  weeks ahead (sunflower 8–10 weeks). Mustard seed is bought at harvest, March to May, for
  most of the year. Rice bran oil is bought monthly from mills.
- **Prices.** Shelf prices follow raw-material costs within 2–4 weeks. Distributors stock up
  before an announced price rise and run stock down afterwards. Sweet shops and caterers
  switch between palmolein, soybean and vanaspati on price.
- **Seasons.**
  - **Mustard oil** sells most in winter (November to February), when North Indian households
    cook with it more.
  - **Festival buying** happens in the three weeks before Diwali and in the days before Holi
    and Dussehra.
  - **Caterer tins** follow the wedding season, with a pause from mid-December to mid-January.
  - **Summer and the monsoon** (June to August) are slow.

## What happened in the last three years

What the planning manager remembers. All of it is visible in the ERP history.

| When | What | Effect on sales |
| --- | --- | --- |
| Jan 2024 | Severe cold wave in North India (Delhi's coldest January in years) | Mustard oil sold well above normal |
| May–Jun 2024 | Extreme heat in Delhi and Rajasthan | Caterer and sweet-shop tins softer |
| 14 Sep 2024 | Government raised the duty on crude palm, soybean and sunflower oil from 0% to 20% | Distributors bought ahead in September. Shelf prices rose 10–15%. Imported oils sold less for about six months, with some buyers moving to mustard oil |
| 31 May 2025 | Duty on crude edible oils cut from 20% to 10% | Prices eased from June 2025, and imported oils recovered |
| 31 Oct 2024, 20 Oct 2025, 8 Nov 2026 | Diwali dates | Festive buying follows the date: mostly in October each year, starting a little later in 2026 |

## What management needs from the forecast

1. How much crude soybean, palm and sunflower oil to book for the next two months.
2. How much mustard seed to buy at the March–May 2027 harvest.
3. When to revise the price list.
4. How much stock to build for weddings and Holi (22 March 2027).

## What is real and what is invented

| Real | Invented |
| --- | --- |
| New Delhi monthly temperatures, Nov 2023 – Oct 2026 (Open-Meteo historical archive) | The company, brands, customers, plants and people |
| Festival dates (Diwali, Dussehra, Holi) | Volumes, prices, growth rates, noise |
| Import duty hike on 14 Sep 2024 and cut on 31 May 2025 | How strongly sales react: mustard +3.5% per °C colder winter, price elasticity −0.6 (retail) and −0.8 (tins) |

Nothing after October 2026 is in the data. What happens next is for the AI to find in live news,
weather outlooks and NOAA bulletins.
