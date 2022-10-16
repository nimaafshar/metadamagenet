# Evaluation Results

## Localization Results

**validation set**

path in dataset : `/test/`

<table>
<thead>
    <tr>
        <td colspan="1">model</td>
        <td colspan="1">version</td>
        <td colspan="8">Localization Score</td>
    </tr>
    <tr>
        <td colspan="2">seed</td>
        <td colspan="2">0</td>
        <td colspan="2">1</td>
        <td colspan="2">2</td>
        <td colspan="2">mean agg.</td>
    </tr>
    <tr>
        <td colspan="2"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td colspan="2"> - </td>
    </tr>
</thead>
<tbody>
<tr>
  <td>Resnet34Unet</td>
  <td>1</td>

  <td>0.6542</td>
  <td>0.6590</td>

  <td>0.6664</td>
  <td>0.6768</td>

  <td>0.6812</td>
  <td>0.6858</td>

  <td colspan="2">0.6720</td>
</tr>
<tr>
<td>SeResnext50Unet</td>
<td>tuned</td>

<td>0.6957</td>
<td>0.6967</td>

<td>0.6916</td>
<td>0.6971</td>

<td>0.6981</td>
<td>0.7027</td>

<td>0.6998</td>
</tr>
<tr>
<td>Dpn92Unet</td>
<td>tuned</td>

<td>0.6776</td>
<td>0.6830</td>

<td>0.6335</td>
<td>0.6322</td>

<td>0.6662</td>
<td>0.6714</td>

<td colspan="2">0.6637</td>
</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>1</td>

  <td>0.7261</td>
  <td>0.7302</td>

  <td>0.7100</td>
  <td>0.7163</td>

  <td>0.7217</td>
  <td>0.7252</td>

  <td colspan="2">0.7264</td>
</tr>
</tbody>
</table>

**test set**

path in dataset : `/hold/`

<table>
<thead>
    <tr>
        <td colspan="1">model</td>
        <td colspan="1">version</td>
        <td colspan="8">Localization Score</td>
    </tr>
    <tr>
        <td colspan="2">seed</td>
        <td colspan="2">0</td>
        <td colspan="2">1</td>
        <td colspan="2">2</td>
        <td colspan="2">mean agg.</td>
    </tr>
    <tr>
        <td colspan="2"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td colspan="2"> - </td>
    </tr>
</thead>
<tr>
  <td>Resnet34Unet</td>
  <td>1</td>

  <td>0.6590</td>
  <td>0.6643</td>

  <td>0.6690</td>
  <td>0.6799</td>

  <td>0.6839</td>
  <td>0.6903</td>

  <td colspan="2">0.6772</td>
</tr>
<tr>
  <td>SeResnext50Unet</td>
  <td>tuned</td>

  <td>0.6963</td>
  <td>0.7002</td>

  <td>0.7036</td>
  <td>0.7074</td>

  <td>0.7084</td>
  <td>0.7087</td>

  <td colspan="2">0.7088</td>
</tr>
<tr>
<td>Dpn92Unet</td>
<td>tuned</td>

<td>0.6796</td>
<td>0.6849</td>

<td>0.6297</td>
<td>0.6335</td>

<td>0.6708</td>
<td>0.6722</td>

<td colspan="2">0.6597</td>

</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>1</td>

  <td>0.7348</td>
  <td>0.7393</td>

  <td>0.7253</td>
  <td>0.7319</td>

  <td>0.7326</td>
  <td>0.7360</td>

  <td colspan="2">0.7409</td>
</tr>
</table>

## Classification Results

**validation set**

path in dataset : `/test/`

<table>
<thead>
    <tr>
        <td colspan="1">model</td>
        <td colspan="1">version</td>
        <td colspan="8">Classification Score</td>
    </tr>
    <tr>
        <td colspan="2">seed</td>
        <td colspan="2">0</td>
        <td colspan="2">1</td>
        <td colspan="2">2</td>
        <td colspan="2">mean agg.</td>
    </tr>
    <tr>
        <td colspan="2"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td colspan="2"> - </td>
    </tr>
</thead>
<tbody>
<tr>
  <td>Resnet34Unet</td>
  <td>tuned</td>

  <td>0.1119</td>
  <td>0.0831</td>

  <td>0.1264</td>
  <td>0.0997</td>

  <td>0.1324</td>
  <td>0.1082</td>

  <td colspan="2">0.0832</td>
</tr>
<tr>
<td>SeResnext50Unet</td>
<td>tuned</td>

<td>0.6397</td>
<td>0.6347</td>

<td>0.6012</td>
<td>0.5991</td>

<td>0.6271</td>
<td>0.6361</td>

<td>0.6301</td>
</tr>
<tr>
<td>Dpn92Unet</td>
<td>tuned</td>

<td>0.6387</td>
<td>0.6441</td>

<td>0.5869</td>
<td>0.5813</td>

<td>0.6075</td>
<td>0.6138</td>

<td colspan="2">0.6258</td>
</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>tuned</td>

  <td>0.6684</td>
  <td>0.6722</td>

  <td>0.5889</td>
  <td>0.6123</td>

  <td>0.6520</td>
  <td>0.6479</td>

  <td colspan="2">0.6596</td>
</tr>
</tbody>
</table>

**test set**

path in dataset : `/hold/`

<table>
<thead>
    <tr>
        <td colspan="1">model</td>
        <td colspan="1">version</td>
        <td colspan="8">Classification Score</td>
    </tr>
    <tr>
        <td colspan="2">seed</td>
        <td colspan="2">0</td>
        <td colspan="2">1</td>
        <td colspan="2">2</td>
        <td colspan="2">mean agg.</td>
    </tr>
    <tr>
        <td colspan="2"> TTA </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td> - </td>
        <td> + </td>
        <td colspan="2"> - </td>
    </tr>
</thead>
<tbody>
<tr>
  <td>Resnet34Unet</td>
  <td>tuned</td>

  <td>0.1090</td>
  <td>0.0806</td>

  <td>0.1466</td>
  <td>0.1174</td>

  <td>0.1314</td>
  <td>0.1101</td>

  <td colspan="2">0.0860</td>
</tr>
<tr>
<td>SeResnext50Unet</td>
<td>tuned</td>

<td>0.6164</td>
<td>0.6152</td>

<td>0.6135</td>
<td>0.6069</td>

<td>0.6319</td>
<td>0.6422</td>

<td colspan="2">0.6360</td>
</tr>
<tr>
<td>Dpn92Unet</td>
<td>tuned</td>

<td>0.6564</td>
<td>0.6657</td>

<td>0.6233</td>
<td>0.6343</td>

<td>0.6246</td>
<td>0.6252</td>

<td colspan="2">0.6460</td>
</tr>
<tr>
  <td>SeNet154Unet</td>
  <td>tuned</td>

  <td>0.6916</td>
  <td>0.7034</td>

  <td>0.6216</td>
  <td>0.6342</td>

  <td>0.6868</td>
  <td>0.6949</td>

  <td colspan="2">0.6954</td>
</tr>
</tbody>
</table>
