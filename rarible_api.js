const YOUR_API_KEY = 'sim_Vgz1JuOxeJzDt50AJ9FxDPrFY4NFp2dC'



// // Get transaction history for risk analysis
// const walletAddress = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"
// const response = await fetch(`https://api.sim.dune.com/v1/evm/activity/${walletAddress}?limit=25`, {
//     headers: {
//         'X-Sim-Api-Key': 'sim_Vgz1JuOxeJzDt50AJ9FxDPrFY4NFp2dC',
//         'Content-Type': 'application/json'
//     }
// });

// const data = await response.json();
// console.log(data)
// const activities = data.activity || [];

// OUTPUT - 
//  {
//       chain_id: 1,
//       block_number: 22961942,
//       block_time: '2025-07-20T17:37:35+00:00',
//       tx_hash: '0xcaff866e7dfdc6c4ba706d004eb22ed2e76d8e45b8c2ba1c87237b8c11231791',
//       tx_from: '0x76868cc6daac92c9c2368af2533c881855484408',
//       tx_to: '0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb',
//       tx_value: '0',
//       type: 'call',
//       call_type: 'incoming',
//       from: '0x76868cc6daac92c9c2368af2533c881855484408',
//       value: '0',
//       data: '0xc44193c30000000000000000000000000000000000000000000000000000000000000d16000000000000000000000000000000000000000000000002a
// 3b5a10781c50000',
//       contract_metadata: [Object],
//       decoded: [Object]
//     },
//     {
//       chain_id: 137,
//       block_number: 74197404,
//       block_time: '2025-07-20T17:02:33+00:00',
//       tx_hash: '0x6885ec7eb40d14be7d38d8597fddffac4adc20c7531b9e20d51af12192cde197',
//       tx_from: '0x081b6355a6fd09367b68c54be1848bd30e3d8dff',
//       tx_to: '0x7d496ee913ce67e9b9dc9a929bb65a101ec74642',
//       tx_value: '0',
//       type: 'receive',
//       asset_type: 'erc20',
//       token_address: '0x7d496ee913ce67e9b9dc9a929bb65a101ec74642',
//       from: '0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245',
//       value: '1000000000000000000000',
//       token_metadata: [Object]
//     },
//     {
//       chain_id: 1,
//       block_number: 22961562,
//       block_time: '2025-07-20T16:21:11+00:00',
//       tx_hash: '0xfb2146bb0dfacfa13c361aa7fe77a888511765270911103d78e2418553609775',
//       tx_from: '0x4d60c4016f160d7e867ae14e03d631c2aaa146d4',
//       tx_to: '0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb',
//       tx_value: '0',
//       type: 'call',
//       call_type: 'incoming',
//       from: '0x4d60c4016f160d7e867ae14e03d631c2aaa146d4',
//       value: '0',
//       data: '0xc44193c300000000000000000000000000000000000000000000000000000000000009930000000000000000000000000000000000000000000000025
// 4beb02d1dcc0000',
//       contract_metadata: [Object],
//       decoded: [Object]
//     },











// // Analyze NFT collection holder distribution
// const contractAddress = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"
// const response = await fetch(`https://api.sim.dune.com/v1/evm/token-holders/1/${contractAddress}?limit=100`, {
//     headers: {
//         'X-Sim-Api-Key': 'sim_Vgz1JuOxeJzDt50AJ9FxDPrFY4NFp2dC',
//         'Content-Type': 'application/json'
//     }
// });

// const data = await response.json();
// console.log(data)


//   token_address: '0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb',
//   chain_id: 1,
//   holders: [
//     {
//       wallet_address: '0xb7f7f6c52f2e2fdb1963eab30438024864c313f6',
//       balance: '593',
//       first_acquired: '2020-09-09T11:17:09+00:00',
//       has_initiated_transfer: false
//     },
//     {
//       wallet_address: '0xa858ddc0445d8131dac4d1de01f834ffcba52ef1',
//       balance: '414',
//       first_acquired: '2022-03-16T19:57:01+00:00',
//       has_initiated_transfer: false
//     },


// to check how many differnt wallet address they have from 50 and how many has has_initiated_transfer






// // Analyze wallet portfolio for context
// const walletAddress = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"

// const response = await fetch(`https://api.sim.dune.com/v1/evm/balances/${walletAddress}?metadata=url,logo`, {
//     headers: {
//         'X-Sim-Api-Key': YOUR_API_KEY,
//         'Content-Type': 'application/json'
//     }
// });

// const data = await response.json();
// console.log(data)

// Risk data you get:

// Portfolio diversification metrics
// USD valuations of holdings
// Token liquidity data (pool_size, low_liquidity)
// Multi-chain exposure analysis



//  {
//       chain: 'bnb',
//       chain_id: 56,
//       address: '0xf0229afd4fcb5168790b39049eb055a1e68488af',
//       amount: '73529550017999998737652383744',
//       symbol: 'WNCG',
//       name: 'Wrapped NCG',
//       decimals: 18,
//       price_usd: 2.2517248442000104e-8,
//       value_usd: 1655.6831455837794,
//       pool_size: 0.005627792185013378,
//       low_liquidity: true
//     },
//     {
//       chain: 'bnb',
//       chain_id: 56,
//       address: '0x304582e2d68a64389662278495c88851eba47fc7',
//       amount: '559385986321999964327192297472',
//       symbol: 'POLO',
//       name: 'Polkaplay',
//       decimals: 18,
//       price_usd: 2.8343877756639976e-9,
//       value_usd: 1585.516801508825,
//       pool_size: 0.004086184530632216,
//       low_liquidity: true
//     },
//     {
//       chain: 'bnb',
//       chain_id: 56,
//       address: '0x7d1d28a40075d65a0be769cba109a27404f861ad',
//       amount: '176063355296000015979920752640',
//       symbol: 'THG',
//       name: 'Thetan Gem',
//       decimals: 18,
//       price_usd: 7.865692634851328e-9,
//       value_usd: 1384.8602370189599,
//       pool_size: 0.005815544496602645,
//       low_liquidity: true
//     },
//     {
//       chain: 'bnb',
//       chain_id: 56,
//       address: '0x09e8cc56789173fe34b1ea0ea743fe4297e05ba8',
//       amount: '147043980339000008511938625536',
//       symbol: 'KABY',
//       name: 'Kaby Arena',
//       decimals: 18,
//       price_usd: 7.612763619501755e-9,
//       value_usd: 1119.4110639914704,
//       pool_size: 0.0062015847812294965,
//       low_liquidity: true
//     },
//     {
//       chain: 'ethereum',
//       chain_id: 1,
//       address: '0x6c267d4a8a8158cd6b0e4edd010b1e4d1dc04f61',
//       amount: '4500000000000000000',
//       symbol: 'SMAR',
//       name: 'ShibaMars',
//       decimals: 9,
//       price_usd: 2.3168272560084333e-7,
//       value_usd: 1042.572265203795,
//       pool_size: 2047.4047392838922,
//       low_liquidity: true
//     },













// Get detailed token information for risk assessment
// const tokenAddress = "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb"
// const response = await fetch(`https://api.sim.dune.com/v1/evm/token-info/${tokenAddress}?chain_ids=1`, {
//     headers: {
//         'X-Sim-Api-Key': YOUR_API_KEY,
//         'Content-Type': 'application/json'
//     }
// });

// const data = await response.json();
// console.log(data)
