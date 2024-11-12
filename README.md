<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <h1>🏆1st place solution, project_X MISIS AsoFast X Gazprombank Hackathon🏆</h1>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="классификатор-типа-личности">Классификатор типа личности</h1>
<p>Алгоритм, который автоматически анализирует данные, полученные со страницы пользователя в социальной сети. На основании выбранной типологии алгоритм определяет тип личности пользователя и предоставляет соответствующую информацию.</p>
<h2 id="типология">Типология</h2>

<table>
<thead>
<tr>
<th></th>
<th>Социальное направление</th>
<th>Способ принятия решений</th>
<th>Когнитивный стиль</th>
</tr>
</thead>
<tbody>
<tr>
<td>1 Тип</td>
<td>Экстраверт</td>
<td>Аналитический</td>
<td>Ригидность</td>
</tr>
<tr>
<td>2 Тип</td>
<td>Интроверт</td>
<td>Чувственный</td>
<td>Гибкость</td>
</tr>
</tbody>
</table><p><strong>Итоговый результат:</strong> 1/2 + Э/И + А/Ч + Р/Г</p>
<blockquote>
<p>Возрастная категория + соц. напр. + сп. прин. реш-ий + когн. стиль</p>
</blockquote>
<h2 id="подготовка-данных">Подготовка данных</h2>
<img  src="https://github.com/SoberSinceToday/AI_Hack_AsoFast_Gazprombank/blob/d64b9cd74eb1108232c55ec52a4698c2ea78bc0a/utils_for_readme/image3.png"/>
<p>Берутся <strong>уже имеющиеся данные</strong>, на их основе <strong>добавочно создаются новые признаки</strong>.</p>
<blockquote>
<p>Например, среднее кол-во лайков на посте, использование характерных слов в процессе интернет-активности, стиль речи и т.п.</p>
</blockquote>
<p><strong>Финальный вид данных:</strong></p>
<ul>
<li>Первая целевая:  avg лайков на фото, кол-во фото, кол-во постов владельца, кол-во постов друзей, кол-во групп, кол-во удаленных постов, кол-во групп знакомств/чатов, различия городов друзей, кол-во друзей</li>
<li>Вторая целевая: возраст, пол, разница в интересах пользователя</li>
<li>Третья целевая: усредненные эмбеддинги групп пользователя, avg расстояние между группами пользователя, усредненные эмбеддинги постов пользователя, avg расстояние между постами пользователя</li>
</ul>
<p><strong>Работа с данными</strong></p>
<ul>
<li>Первая целевая: берутся указанные колонки, по ним обучается K-mean, затем, объекты с неявно определенной целевой(например, с разбиением шансов 49% и 51% соотв.) отбрасываются, остальные остаются</li>
<li>Вторая целевая: указанные выше колонки берутся, разница в интересах считается как разность упоминания разных категорий в таких же таблицах. K-mean распределяет кластеры.</li>
<li>Третья целевая: текстовое описание постов и групп пользователя переводятся в эмбеддинги, avg расстояние считается как разность упоминания разных категорий в таких же таблицах. K-mean распределяет кластеры.</li>
</ul>
<h2 id="telegram-бот">Telegram бот</h2>
<h3 id="блок-схема-концепции-использования-бота">Блок-схема концепции использования бота</h3>
<img  src="https://github.com/SoberSinceToday/AI_Hack_AsoFast_Gazprombank/blob/d64b9cd74eb1108232c55ec52a4698c2ea78bc0a/utils_for_readme/image4.png"/>
<p>Телеграм-бот используется в кач-ве GUI для более комфортного использования/тестирования алгоритма.</p>
<p>Для написания используется фреймворк Aiogram,</p>
<table>
    <tr>
        <td><img src="https://raw.githubusercontent.com/SoberSinceToday/AI_Hack_AsoFast_Gazprombank/main/utils_for_readme/image.png"/></td>
        <td><img src="https://raw.githubusercontent.com/SoberSinceToday/AI_Hack_AsoFast_Gazprombank/main/utils_for_readme/image2.png"/></td>
    </tr>
</table>
</div>

<h2>Описание репозитория🥕</h2>


<body>
    <ul style="font-size: 34px;">
        <li>🤖bot - файлы telegram-бота🤖</li>
        <li>💩ipynb_solution - "грязное решение"💩</li>
        <li>🧽py solution - решение чуть почище🧽</li>
        <li>🔧utils_for_readme - вспомогательные файлы для README🔧</li>
        <li>📤 ans, answer - файлы заливки и компановки решения соотв.📤</li>
        <li>🎤projectX misis.pdf - питч-презентация🎤</li>
    </ul>
</body>
</body>
</html>